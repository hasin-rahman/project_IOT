// Pure Kotlin/JVM module: no Android classes, so the same code runs on the phone and in tests.
plugins {
    id("org.jetbrains.kotlin.jvm")
    application
}

application {
    // Virtual phone / self test harness:  ./gradlew :core:run --args="selftest"
    mainClass.set("com.fedpose.cli.VirtualPhoneKt")
}
