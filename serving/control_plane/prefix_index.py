from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixScope:
    model_id: str
    model_revision: str
    tokenizer_revision: str
    rope_config: str


@dataclass(frozen=True)
class PrefixMatch:
    source_slot: int
    prefix_length: int
    token_ids: tuple[int, ...]
    fingerprint: str


@dataclass(frozen=True)
class _PrefixEntry:
    scope: PrefixScope
    token_ids: tuple[int, ...]
    source_slot: int
    fingerprint: str


class PrefixCacheIndex:
    """
    In-process token prefix index.

    Hashes are used to locate candidates efficiently. Full scope and token
    equality are always checked before returning a match.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, list[_PrefixEntry]] = {}
        self._entry_count = 0

        self._lookups = 0
        self._hits = 0
        self._misses = 0
        self._reused_tokens = 0
        self._evictions = 0

    def register(
        self,
        *,
        scope: PrefixScope,
        token_ids: list[int] | tuple[int, ...],
        source_slot: int,
    ) -> None:
        normalized_tokens = self._normalize_token_ids(
            token_ids,
            allow_empty=False,
        )
        self._validate_source_slot(source_slot)

        fingerprint = self._fingerprint(scope, normalized_tokens)
        entry = _PrefixEntry(
            scope=scope,
            token_ids=normalized_tokens,
            source_slot=source_slot,
            fingerprint=fingerprint,
        )

        bucket = self._buckets.setdefault(fingerprint, [])

        for index, existing_entry in enumerate(bucket):
            if (
                existing_entry.scope == scope
                and existing_entry.token_ids == normalized_tokens
            ):
                bucket[index] = entry
                return

        bucket.append(entry)
        self._entry_count += 1

    def lookup(
        self,
        *,
        scope: PrefixScope,
        token_ids: list[int] | tuple[int, ...],
    ) -> PrefixMatch | None:
        normalized_tokens = self._normalize_token_ids(
            token_ids,
            allow_empty=True,
        )
        self._lookups += 1

        if not normalized_tokens:
            self._misses += 1
            return None

        fingerprints = self._prefix_fingerprints(scope, normalized_tokens)

        for prefix_length in range(len(normalized_tokens), 0, -1):
            fingerprint = fingerprints[prefix_length - 1]
            prefix_tokens = normalized_tokens[:prefix_length]

            for entry in self._buckets.get(fingerprint, []):
                if entry.scope == scope and entry.token_ids == prefix_tokens:
                    self._hits += 1
                    self._reused_tokens += prefix_length
                    return PrefixMatch(
                        source_slot=entry.source_slot,
                        prefix_length=prefix_length,
                        token_ids=entry.token_ids,
                        fingerprint=entry.fingerprint,
                    )

        self._misses += 1
        return None

    def remove_slot(self, *, source_slot: int) -> None:
        self._validate_source_slot(source_slot)

        removed_entries = 0

        for fingerprint in list(self._buckets):
            bucket = self._buckets[fingerprint]
            retained_entries = [
                entry
                for entry in bucket
                if entry.source_slot != source_slot
            ]

            removed_entries += len(bucket) - len(retained_entries)

            if retained_entries:
                self._buckets[fingerprint] = retained_entries
            else:
                del self._buckets[fingerprint]

        self._entry_count -= removed_entries
        self._evictions += removed_entries

    def stats(self) -> dict[str, int]:
        return {
            "lookups": self._lookups,
            "hits": self._hits,
            "misses": self._misses,
            "reused_tokens": self._reused_tokens,
            "registered_entries": self._entry_count,
            "evictions": self._evictions,
        }

    @staticmethod
    def _validate_source_slot(source_slot: int) -> None:
        if not isinstance(source_slot, int) or isinstance(source_slot, bool):
            raise TypeError("source_slot must be an integer")

        if source_slot < 0:
            raise ValueError(
                f"source_slot must be non-negative, got {source_slot}"
            )

    @staticmethod
    def _normalize_token_ids(
        token_ids: list[int] | tuple[int, ...],
        *,
        allow_empty: bool,
    ) -> tuple[int, ...]:
        normalized_tokens = tuple(token_ids)

        if not allow_empty and not normalized_tokens:
            raise ValueError("token_ids must not be empty")

        for token_id in normalized_tokens:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("token_ids must contain integers")

            if not 0 <= token_id < 2**64:
                raise ValueError(
                    f"token_id must be in [0, 2**64), got {token_id}"
                )

        return normalized_tokens

    def _fingerprint(
        self,
        scope: PrefixScope,
        token_ids: tuple[int, ...],
    ) -> str:
        return self._prefix_fingerprints(scope, token_ids)[-1]

    def _prefix_fingerprints(
        self,
        scope: PrefixScope,
        token_ids: tuple[int, ...],
    ) -> list[str]:
        hasher = hashlib.blake2b(digest_size=16)

        for field in (
            scope.model_id,
            scope.model_revision,
            scope.tokenizer_revision,
            scope.rope_config,
        ):
            encoded_field = field.encode("utf-8")
            hasher.update(len(encoded_field).to_bytes(4, "little"))
            hasher.update(encoded_field)

        fingerprints: list[str] = []

        for token_id in token_ids:
            hasher.update(token_id.to_bytes(8, "little", signed=False))
            fingerprints.append(hasher.hexdigest())

        return fingerprints