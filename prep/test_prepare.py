from prepare import clean, build


def test_clean_drops_stopwords_and_digits():
    assert clean(["the", "cat", "one", "sat", "nine", "mat"], {"the"}) == ["cat", "sat", "mat"]


def test_build_orders_by_count_then_alpha_and_caps():
    vocab, ids = build(["b", "a", "c", "b", "a", "d"], cap=2)
    assert vocab == ["a", "b"]
    assert ids.dtype.name == "int32"
    assert list(ids) == [1, 0, -1, 1, 0, -1]


if __name__ == "__main__":
    test_clean_drops_stopwords_and_digits()
    test_build_orders_by_count_then_alpha_and_caps()
    print("ok")
