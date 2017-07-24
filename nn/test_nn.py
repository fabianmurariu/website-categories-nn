from predict_cat import train, relational
import numpy as np


def test_groupiterables_in_chunks():
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
    actual = train.load_class_weights('test_resources/class_weights.jl', ["bikes", "fashion", "health", "property"])
    assert actual == {0: 27.924082140634724, 1: 8.422297297297296, 2: 3.32227733767676, 3: 3.5400757336699273}


def test_data_point():
    actual = train.data_point("""{"paddedWords":[3540,14221,3540,8227,17329,5953,17826],"category":[1,0]}""")
    assert np.array_equal(actual[0], np.array([3540, 14221, 3540, 8227, 17329, 5953, 17826]))
    assert np.array_equal(actual[1], np.array([1, 0]))


def test_create_batch():
    x_actual, y_actual = train.create_batch([(1, 2), (3, 4)])
    assert np.array_equal(x_actual, np.array([1, 3]))
    assert np.array_equal(y_actual, np.array([2, 4]))


def test_load_clever_questions():
    json_lines = [js for js in relational.load_clever_questions('test_resources/clever/test.jl', False)]
    assert len(json_lines) == 20


def test_generate_one_hot_encoding_and_word_index_for_answers():
    expected_word_index = {u'blue': 0, u'large': 5, u'cylinder': 2, u'no': 3, u'metal': 4, u'1': 1, u'0': 6, u'2': 7,
                           u'yes': 8}
    word_index, enc = relational.ohc_word_index_answers('test_resources/clever/test.jl')
    assert word_index == expected_word_index
    actual = enc.transform([[1], [3], [5]]).toarray()
    assert np.array_equal(actual, np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                            [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 1., 0., 0., 0.]]))


def test_generate_word_index_questions():
    actual = relational.word_index_questions('test_resources/clever/test.jl')
    assert actual['right'] == 54
    assert actual['cube'] == 56
    assert actual['are'] == 1
    actual_size = len(actual)
    assert actual_size == 67


def test_load_x_y_questions():
    actual = list(relational.load_x_y_questions('test_resources/clever/test.jl', 45, 5))
    actual_size = len(actual)
    assert actual_size == 4
    x_shape = actual[0][0].shape
    y_shape = actual[0][1].shape
    assert x_shape == (5, 45)
    assert y_shape == (5, 9)
