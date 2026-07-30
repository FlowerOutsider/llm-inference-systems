from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

from serving.control_plane.prefix_cache_coordinator import PrefixCacheCoordinator
from serving.control_plane.prefix_index import PrefixMatch, PrefixScope
from serving.data_plane.paged_kv_cache import PagedKVCache


@dataclass(frozen=True)
class PrefixAdmissionResult:
    admitted: bool
    reason: str


@dataclass(frozen=True)
class PrefixCacheEntry:
    scope: PrefixScope
    token_ids: tuple[int, ...]
    source_slot: int


class PrefixCacheManager:
    """Owns prefix-cache admission, eviction, and policy-level statistics."""

    def __init__(
        self,
        *,
        cache: PagedKVCache,
        coordinator: PrefixCacheCoordinator,
        min_prefix_tokens: int,
        max_entries: int,
        min_free_blocks: int,
    ) -> None:
        if min_prefix_tokens <= 0:
            raise ValueError("min_prefix_tokens must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if min_free_blocks < 0:
            raise ValueError("min_free_blocks must be non-negative")

        self._cache = cache
        self._coordinator = coordinator
        self._min_prefix_tokens = min_prefix_tokens
        self._max_entries = max_entries
        self._min_free_blocks = min_free_blocks

        # OrderedDict 从左到右表示 LRU 到 MRU。
        self._entries: OrderedDict[
            tuple[PrefixScope, tuple[int, ...]],
            PrefixCacheEntry,
        ] = OrderedDict()
        self._source_to_key: dict[int, tuple[PrefixScope, tuple[int, ...]]] = {}

        self._admissions = 0
        self._rejected_short_prefixes = 0
        self._rejected_capacity = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def admit(
        self,
        *,
        scope: PrefixScope,
        token_ids: Sequence[int],
        source_slot: int,
    ) -> PrefixAdmissionResult:
        normalized_token_ids = tuple(token_ids)

        if len(normalized_token_ids) < self._min_prefix_tokens:
            self._rejected_short_prefixes += 1
            return PrefixAdmissionResult(
                admitted=False,
                reason="prefix_too_short",
            )

        key = (scope, normalized_token_ids)

        existing_key = self._source_to_key.get(source_slot)
        if existing_key is not None and existing_key != key:
            raise ValueError(
                f"source slot {source_slot} is already owned by another prefix entry"
            )

        if key in self._entries:
            existing_entry = self._entries[key]

            if existing_entry.source_slot != source_slot:
                self._evict_key(key)

            else:
                self._touch(key)
                return PrefixAdmissionResult(
                    admitted=True,
                    reason="already_cached",
                )

        self._evict_until_admissible()

        if (
            len(self._entries) >= self._max_entries
            or self._cache.free_block_count < self._min_free_blocks
        ):
            self._rejected_capacity += 1
            return PrefixAdmissionResult(
                admitted=False,
                reason="insufficient_capacity",
            )

        self._coordinator.publish_prefix(
            scope=scope,
            token_ids=normalized_token_ids,
            source_slot=source_slot,
        )

        entry = PrefixCacheEntry(
            scope=scope,
            token_ids=normalized_token_ids,
            source_slot=source_slot,
        )
        self._entries[key] = entry
        self._source_to_key[source_slot] = key
        self._admissions += 1

        return PrefixAdmissionResult(
            admitted=True,
            reason="admitted",
        )

    def attach_longest_prefix(
        self,
        *,
        scope: PrefixScope,
        token_ids: Sequence[int],
        target_slot: int,
    ) -> PrefixMatch | None:
        match = self._coordinator.attach_longest_prefix(
            scope=scope,
            token_ids=token_ids,
            target_slot=target_slot,
        )

        if match is None:
            self._misses += 1
            return None

        key = self._source_to_key.get(match.source_slot)
        if key is None:
            raise RuntimeError(
                f"prefix source slot {match.source_slot} is not owned by this manager"
            )

        self._touch(key)
        self._hits += 1
        return match

    def evict_lru(self) -> PrefixCacheEntry | None:
        if not self._entries:
            return None

        key = next(iter(self._entries))
        return self._evict_key(key)

    def cached_source_slots(self) -> tuple[int, ...]:
        return tuple(entry.source_slot for entry in self._entries.values())

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "admissions": self._admissions,
            "rejected_short_prefixes": self._rejected_short_prefixes,
            "rejected_capacity": self._rejected_capacity,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "free_blocks": self._cache.free_block_count,
            "min_free_blocks": self._min_free_blocks,
        }

    def _evict_until_admissible(self) -> None:
        while self._entries and (
            len(self._entries) >= self._max_entries
            or self._cache.free_block_count < self._min_free_blocks
        ):
            self.evict_lru()

    def _evict_key(
        self,
        key: tuple[PrefixScope, tuple[int, ...]],
    ) -> PrefixCacheEntry:
        entry = self._entries.pop(key)
        self._source_to_key.pop(entry.source_slot)

        # Coordinator 会先删除 PrefixIndex 条目，再释放 source slot；
        # 对仍共享该 block 的目标请求，PagedKVCache 的引用计数会保证安全。
        self._coordinator.release_slots([entry.source_slot])

        self._evictions += 1
        return entry

    def _touch(self, key: tuple[PrefixScope, tuple[int, ...]]) -> None:
        self._entries.move_to_end(key)