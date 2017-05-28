import org.tensorflow.{SavedModelBundle, Tensor}

/**
  * Load a tensorflow model produced with Keras
  * and predict one of the categories
  */
object TfModelLoader {


  def main(args: Array[String]): Unit = {

    val home = System.getProperty("user.home")
    val smb = SavedModelBundle.load(home + "/ml-work/dmoz/model-tf-serve4", "serve")
    val session = smb.session()

    val inputSize = 1000
    val inputTensor = Tensor.create(sample)

    val result: Tensor = session
      .runner()
      .feed("input_1", inputTensor)
      .fetch("dense_2/Softmax")
      .run().get(0)

    val outputSize = 20
    val m: Array[Array[Float]] = Array.fill[Array[Float]](1){ new Array[Float](outputSize) }

    val matrix = result.copyTo(m)

    println(matrix.toVector.map(_.toVector))
  }

  def sample: Array[Array[Int]] = Array(Array[Int](16170, 10206, 19649, 1736, 8026, 5146, 13548, 16170, 10206,
    7591, 1606, 8497, 716, 2448, 18810, 8274, 1736, 619,
    8250, 15436, 13719, 13158, 549, 709, 10219, 18672, 5033,
    18460, 12611, 159, 16485, 13158, 18842, 8480, 10206, 4075,
    17026, 3078, 19147, 7204, 18810, 2446, 3289, 1342, 18810,
    2950, 15746, 19645, 19378, 2476, 1736, 19649, 6899, 1736,
    8026, 19649, 9810, 287, 16170, 10206, 19649, 1736, 8026,
    1863, 6400, 13158, 549, 10188, 583, 11275, 1311, 8278,
    12839, 4597, 13158, 2448, 1466, 6514, 709, 7776, 16878,
    11822, 17083, 6514, 10205, 11717, 10368, 583, 6400, 3425,
    14905, 10668, 1865, 17817, 2668, 364, 18810, 6082, 2668,
    364, 4597, 901, 1863, 7646, 18289, 9784, 1857, 1466,
    19649, 972, 583, 9082, 7646, 15220, 7959, 18289, 9784,
    1857, 1466, 11391, 18708, 9082, 7646, 5740, 1466, 1133,
    8501, 4597, 1507, 13182, 721, 11275, 8480, 18708, 16234,
    17026, 6400, 11631, 18708, 1466, 5470, 8480, 13701, 10054,
    19649, 4222, 3169, 3289, 18810, 7966, 4548, 4938, 17711,
    7915, 2550, 18460, 18460, 2476, 1736, 799, 1736, 19645,
    18810, 18127, 5914, 1928, 17812, 11323, 1652, 7087, 13142,
    2550, 18810, 9799, 12802, 9784, 18626, 2964, 9443, 13700,
    12649, 4440, 13698, 6279, 8480, 18439, 9443, 13158, 549,
    17812, 16170, 9784, 9443, 4597, 10206, 16981, 12754, 15617,
    13701, 4597, 19500, 14073, 3463, 2446, 3289, 2950, 1342,
    18810, 7966, 3289, 2243, 2476, 1736, 10206, 4075, 17026,
    12611, 159, 3078, 19147, 15746, 19649, 6899, 1736, 8026,
    19649, 9810, 287, 7204, 18810, 9886, 16485, 13158, 18842,
    8480, 19645, 2446, 1863, 13380, 15747, 19147, 11900, 10362,
    18708, 4597, 1928, 1857, 8344, 583, 4597, 15747, 2504,
    11996, 17812, 11323, 15515, 3078, 19147, 10023, 583, 18810,
    8840, 13698, 9840, 14535, 4338, 8450, 4938, 17711, 1736,
    13276, 13698, 4597, 508, 17420, 619, 1863, 17122, 13698,
    3169, 18974, 371, 13570, 18810, 5371, 13698, 19649, 16397,
    4761, 18439, 9443, 19649, 1736, 8026, 17812, 16170, 16397,
    17901, 18810, 16887, 9443, 249, 16397, 18065, 18289, 18571,
    10183, 19639, 583, 15850, 5740, 120, 19535, 4816, 7858,
    1466, 1387, 14806, 15897, 8497, 13698, 5957, 19535, 709,
    11422, 11442, 11794, 19535, 1736, 8026, 709, 11422, 11442,
    16017, 19535, 709, 11422, 11442, 17926, 19535, 709, 11422,
    11442, 15140, 19535, 5078, 11337, 15368, 246, 18420, 16890,
    709, 11422, 11442, 9784, 19109, 19535, 709, 11422, 11442,
    6802, 19535, 709, 11422, 11442, 17710, 14109, 19500, 4823,
    13490, 722, 3408, 6899, 2446, 2446, 3289, 1342, 9251,
    2446, 12426, 17996, 1914, 2243, 1599, 17026, 15747, 19147,
    9502, 10818, 382, 17159, 1914, 4999, 2446, 5939, 2226,
    2950, 7581, 947, 947, 15177, 19380, 19123, 10818, 18797,
    9495, 9448, 13700, 1736, 3078, 19147, 15746, 136, 4705,
    16295, 7204, 19645, 19378, 9068, 18224, 2446, 443, 19649,
    1736, 8026, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ))

}