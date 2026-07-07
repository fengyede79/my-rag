from rag_modules.dish_aliases import dish_aliases_for, is_known_alias_target


def test_dish_aliases_for_known_live_failure_names():
    assert dish_aliases_for("番茄炒蛋") == ["西红柿炒鸡蛋", "番茄鸡蛋", "番茄炒鸡蛋"]
    assert dish_aliases_for("凉拌黄瓜") == ["拍黄瓜", "黄瓜"]
    assert dish_aliases_for("可乐鸡翅") == ["鸡翅"]
    assert dish_aliases_for("红烧肉") == ["五花肉"]


def test_dish_aliases_strip_whitespace_and_ignore_unknowns():
    assert dish_aliases_for(" 番茄炒蛋 ") == ["西红柿炒鸡蛋", "番茄鸡蛋", "番茄炒鸡蛋"]
    assert dish_aliases_for("不存在的菜") == []
    assert dish_aliases_for(None) == []


def test_known_alias_target_accepts_original_and_aliases():
    assert is_known_alias_target("番茄炒蛋", "番茄炒蛋") is True
    assert is_known_alias_target("番茄炒蛋", "西红柿炒鸡蛋") is True
    assert is_known_alias_target("番茄炒蛋", "番茄鸡蛋") is True
    assert is_known_alias_target("番茄炒蛋", "鱼香肉丝") is False
    assert is_known_alias_target(None, "番茄炒蛋") is False
