package com.fedpose.core

import java.io.File

/**
 * Loads `<root>/raw/<client>/<split>/<pose>__<user>__<id>.csv` into windowed features.
 * The same layout is used on the phone (app external files dir) and in the Python experiments,
 * so `adb pull` gives Python exactly the data the phone trained on.
 */
object Dataset {
    fun sessionFiles(root: File, client: String, split: String): List<Pair<File, Int>> {
        val dir = File(root, "raw/$client/$split")
        val files = dir.listFiles { f -> f.isFile && f.name.endsWith(".csv") } ?: return emptyList()
        return files.sortedBy { it.name }.mapNotNull { f ->
            val pose = Constants.POSES.indexOf(f.name.removeSuffix(".csv").split("__")[0])
            if (pose < 0) null else f to pose
        }
    }

    fun load(root: File, client: String, split: String, featureSet: String,
             onSkip: (String) -> Unit = {}): LocalData {
        val xs = ArrayList<DoubleArray>()
        val ys = ArrayList<Int>()
        for ((f, pose) in sessionFiles(root, client, split)) {
            if (split == "train" && pose !in Constants.CLIENT_POSES.getValue(client)) {
                onSkip("client $client train split contains pose $pose (${f.name}); ignored (non-IID rule)")
                continue
            }
            for (v in Features.windowFeatures(Features.loadSessionCsv(f), featureSet)) {
                xs.add(v); ys.add(pose)
            }
        }
        return LocalData(xs, ys.toIntArray())
    }
}
