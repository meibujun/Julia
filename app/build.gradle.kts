plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.dagger.hilt.android")
    kotlin("kapt")
}

android {
    namespace = "com.animalgenetics.education"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.animalgenetics.education"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Configure for Chinese market
        resConfigs("en", "zh-rCN", "zh-rTW")
    }

    signingConfigs {
        create("release") {
            // These would be configured in a secure environment, e.g., CI/CD secrets
            // storeFile = file(System.getenv("KEYSTORE_FILE") ?: "keystore.jks")
            // storePassword = System.getenv("KEYSTORE_PASSWORD")
            // keyAlias = System.getenv("KEY_ALIAS")
            // keyPassword = System.getenv("KEY_PASSWORD")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            signingConfig = signingConfigs.getByName("release")
        }
    }

    // Split APKs for size optimization
    splits {
        abi {
            isEnable = true
            reset()
            include("armeabi-v7a", "arm64-v8a", "x86", "x86_64")
            isUniversalApk = true
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
        viewBinding = true
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.3"
    }
}

dependencies {
    // Project Modules
    implementation(project(":core"))
    implementation(project(":domain"))
    implementation(project(":data"))
    implementation(project(":feature:auth"))
    implementation(project(":feature:knowledge-graph"))
    implementation(project(":feature:ai-tutor"))
    implementation(project(":feature:virtual-lab"))
    implementation(project(":feature:assessment"))
    implementation(project(":feature:gamification"))
    implementation(project(":feature:case-studies"))

    // Core Android
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.activity:activity-compose:1.8.2")

    // UI
    implementation("androidx.compose.ui:ui:1.5.4")
    implementation("androidx.compose.material3:material3:1.1.2")
    implementation("com.google.android.material:material:1.11.0")

    // Architecture Components
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    kapt("androidx.room:room-compiler:2.6.1")

    implementation("com.google.dagger:hilt-android:2.48")
    kapt("com.google.dagger:hilt-compiler:2.48")

    // Networking
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")

    // Firebase (using BOM - Bill of Materials)
    implementation(platform("com.google.firebase:firebase-bom:32.7.0"))
    implementation("com.google.firebase:firebase-auth-ktx")
    implementation("com.google.firebase:firebase-firestore-ktx")
    implementation("com.google.firebase:firebase-analytics-ktx")
    implementation("com.google.firebase:firebase-crashlytics-ktx")

    // AI/ML
    implementation("com.google.mediapipe:tasks-genai:0.10.14")

    // Educational Features
    implementation("com.github.PhilJay:MPAndroidChart:v3.1.0")
    implementation("com.jjoe64:graphview:4.2.2")
    // implementation("com.unity3d:unity-classes:2022.3.10f1") // This would require setting up a local repository for the Unity library

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}

// Apply the Hilt plugin
kapt {
    correctErrorTypes = true
}
