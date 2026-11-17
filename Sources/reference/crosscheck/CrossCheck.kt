// Kotlin side of the Python<->Kotlin cross-check. Compiled together with android/app/src/main/java/com/posefl/core.
// Usage: CrossCheck <dir with pose_<i>.csv files>   -> prints features, trained weights and rule-based pose guesses.
import com.posefl.core.FeatureExtractor
import com.posefl.core.Pose
import com.posefl.core.SensorCsv
import com.posefl.core.SoftmaxModel
import java.io.File

fun main(args: Array<String>) {
    val dir = File(args[0])
    val spec = "acc+grav+gyro+rot"
    val X = ArrayList<DoubleArray>()
    val y = ArrayList<Int>()
    for (i in 0 until Pose.COUNT) {
        val rows = SensorCsv.parse(File(dir, "pose_$i.csv").readText())
        val feats = FeatureExtractor.extract(rows, spec)
        X.addAll(feats)
        repeat(feats.size) { y.add(i) }
        val mid = rows[rows.size / 2]
        println("GUESS $i ${Pose.guessFromGravity(mid[4], mid[5], mid[6])?.index ?: -1}")
    }
    for (f in X) println("FEAT " + f.joinToString(" "))
    val m = SoftmaxModel(X[0].size)
    m.train(X, y.toIntArray(), epochs = 3, lr = 0.1, batchSize = 16, l2 = 1e-4, rng = null)
    println("W " + m.W.joinToString(" "))
    println("B " + m.b.joinToString(" "))
    val ev = m.evaluate(X, y.toIntArray())
    println("EVAL ${ev.loss} ${ev.accuracy}")
}
