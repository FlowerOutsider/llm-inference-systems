from __future__ import annotations

from collections.abc import Sequence

from serving.control_plane.prefix_index import (
    PrefixCacheIndex,
    PrefixMatch,
    PrefixScope,
)
from serving.data_plane.paged_kv_cache import PagedKVCache


class PrefixCacheCoordinator:
    """
    Coordinates Prefix Index lifecycle with Paged KV Cache lifecycle.

    The coordinator owns the ordering rule:
    publish only after KV is committed, and remove index entries before the
    corresponding cache slot is released.
    """

    def __init__(
        self,
        *,
        cache: PagedKVCache,
        index: PrefixCacheIndex,
    ) -> None:
        self._cache = cache
        self._index = index

    def allocate_slots(self, count: int = 1) -> list[int]:
        return self._cache.allocate(count)

    def publish_prefix(
        self,
        *,
        scope: PrefixScope,
        token_ids: list[int] | tuple[int, ...],
        source_slot: int,
    ) -> None:
        cache_length = self._cache.length(source_slot)

        if len(token_ids) != cache_length:
            raise ValueError(
                f"token count {len(token_ids)} does not match cache length "
                f"{cache_length} for source slot {source_slot}"
            )

        self._index.register(
            scope=scope,
            token_ids=token_ids,
            source_slot=source_slot,
        )

    def attach_longest_prefix(
        self,
        *,
        scope: PrefixScope,
        token_ids: list[int] | tuple[int, ...],
        target_slot: int,
    ) -> PrefixMatch | None:
        match = self._index.lookup(
            scope=scope,
            token_ids=token_ids,
        )

        if match is None:
            return None

        self._cache.fork_prefix(
            source_slot=match.source_slot,
            target_slot=target_slot,
            prefix_length=match.prefix_length,
        )
        return match

    def release_slots(self, slot_ids: Sequence[int]) -> None:
        normalized_slot_ids = tuple(slot_ids)

        # Remove discoverability first. Afterwards no request can receive a
        # source_slot that is about to return its blocks to the free pool.
        for slot_id in normalized_slot_ids:
            self._index.remove_slot(source_slot=slot_id)

        self._cache.release(normalized_slot_ids)