plugins {
    id("org.jetbrains.kotlin.jvm")
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

dependencies {
    // No Android-specific dependencies
    implementation("org.jetbrains.kotlin:kotlin-stdlib:1.9.10")
}
