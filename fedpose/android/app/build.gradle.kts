plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.fedpose.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.fedpose.app"
        minSdk = 26          // java.util.Base64 and VibrationEffect need API 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation(project(":core"))   // no other dependencies on purpose
}
