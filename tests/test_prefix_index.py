from serving.control_plane.prefix_index import PrefixCacheIndex, PrefixScope


def make_scope(*, model_revision: str = "v1") -> PrefixScope:
    return PrefixScope(
        model_id="demo-llm",
        model_revision=model_revision,
        tokenizer_revision="tokenizer-v1",
        rope_config="rope-theta-10000",
    )


def test_lookup_returns_longest_matching_prefix() -> None:
    index = PrefixCacheIndex()
    scope = make_scope()

    index.register(
        scope=scope,
        token_ids=[10, 11],
        source_slot=3,
    )
    index.register(
        scope=scope,
        token_ids=[10, 11, 12, 13],
        source_slot=7,
    )

    match = index.lookup(
        scope=scope,
        token_ids=[10, 11, 12, 13, 99],
    )

    assert match is not None
    assert match.source_slot == 7
    assert match.prefix_length == 4
    assert match.token_ids == (10, 11, 12, 13)
    assert len(match.fingerprint) > 0

    stats = index.stats()
    assert stats["lookups"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 0
    assert stats["reused_tokens"] == 4
    assert stats["registered_entries"] == 2


def test_scope_isolation_and_slot_removal_prevent_stale_reuse() -> None:
    index = PrefixCacheIndex()
    scope = make_scope()

    index.register(
        scope=scope,
        token_ids=[20, 21, 22],
        source_slot=5,
    )

    assert index.lookup(
        scope=make_scope(model_revision="v2"),
        token_ids=[20, 21, 22, 23],
    ) is None

    index.remove_slot(source_slot=5)

    assert index.lookup(
        scope=scope,
        token_ids=[20, 21, 22, 23],
    ) is None

    stats = index.stats()
    assert stats["lookups"] == 2
    assert stats["hits"] == 0
    assert stats["misses"] == 2
    assert stats["registered_entries"] == 0
    assert stats["evictions"] == 1