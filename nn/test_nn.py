from predict_cat import train
import numpy as np


def test_group_iterables_in_chunks():
    actual = list(train.grouper([1, 2, 3, 4, 5, 6, 7, 8], 3))
    expected = [[1, 2, 3], [4, 5, 6], [7, 8]]
    assert actual == expected


def test_expand_path_to_json_lines():
    actual = list(train.json_lines('test_resources/part*'))
    assert actual == [{'a': 1}, {'b': 2}, {'c': 3}, {'d': 4}]


def test_expand_path_to_text_lines():
    actual = list(train.text_lines('test_resources/part*'))
    assert actual == ["{\"a\": 1}", "{\"b\": 2}", "{\"c\": 3}", "{\"d\": 4}"]


def test_load_class_weights():
    actual = train.class_weights('test_resources/class_weights.jl', ["bikes", "fashion", "health", "property"])
    assert actual == {0: 27.924082140634724, 1: 8.422297297297296, 2: 3.32227733767676, 3: 3.5400757336699273}


def test_data_point():
    actual = train.data_point("""{"paddedWords":[3540,14221,3540,8227,17329,5953,17826],"category":[1,0]}""")
    assert np.array_equal(actual[0], np.array([3540, 14221, 3540, 8227, 17329, 5953, 17826]))
    assert np.array_equal(actual[1], np.array([1, 0]))


def test_create_batch():
    x_actual, y_actual = train.create_batch([(1, 2), (3, 4)])
    assert np.array_equal(x_actual, np.array([1, 3]))
    assert np.array_equal(y_actual, np.array([2, 4]))
