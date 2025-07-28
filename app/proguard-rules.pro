# Proguard rules for common libraries

# Hilt
-keep class dagger.hilt.internal.aggregatedroot.codegen.*
-keep class com.animalgenetics.education.Hilt_GeneticsApplication
-keep class hilt_aggregated_deps.*
-keep @dagger.hilt.InstallIn *
-keep @dagger.hilt.components.SingletonComponent *
-keep @dagger.Module *
-keep @dagger.Provides *
-keep @javax.inject.Inject *
-keep @javax.inject.Singleton *

# Retrofit & OkHttp
-dontwarn retrofit2.Platform
-dontwarn retrofit2.Platform$IOS$MainThreadExecutor
-dontwarn okhttp3.**
-dontwarn okio.**
-dontwarn javax.annotation.**
-keep class retrofit2.** { *; }
-keep class com.google.gson.** { *; }

# Firebase
-keep class com.google.firebase.provider.FirebaseInitProvider
-keep class com.google.android.gms.common.api.internal.TaskApiCall
-keep class com.google.android.gms.common.api.internal.zaar

# Keep our models (data classes) from being obfuscated if they are used with Gson/serialization
-keep class com.animalgenetics.domain.model.** { *; }
-keepclassmembers class com.animalgenetics.domain.model.** { *; }

# Keep Unity callable methods
-keepclassmembers class * {
    @com.animalgenetics.domain.model.UnityCallable <methods>;
}
