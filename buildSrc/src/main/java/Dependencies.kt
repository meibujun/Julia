// This file can be used to manage dependencies in a centralized way.
// For now, it's a placeholder.

object Versions {
    const val kotlin = "1.9.10"
    const val coreKtx = "1.12.0"
    const val appcompat = "1.6.1"
    // ... other versions
}

object Libs {
    const val kotlinStdLib = "org.jetbrains.kotlin:kotlin-stdlib-jdk7:${Versions.kotlin}"
    const val appcompat = "androidx.appcompat:appcompat:${Versions.appcompat}"
    const val coreKtx = "androidx.core:core-ktx:${Versions.coreKtx}"
    // ... other libraries
}
