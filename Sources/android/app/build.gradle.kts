plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.posefl.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.posefl.app"
        minSdk = 26
        targetSdk = 34   // 34 keeps the classic layout below the status bar on Android 15+ (Galaxy S24)
        versionCode = 5
        versionName = "0.5-cloud"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        buildConfig = true
    }
}

// No third-party libraries: only the Android platform APIs (sensors, HttpURLConnection, org.json).
dependencies {
    testImplementation("junit:junit:4.13.2")
}
