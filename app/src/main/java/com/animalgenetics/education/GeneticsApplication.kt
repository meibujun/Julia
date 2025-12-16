package com.animalgenetics.education

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

@HiltAndroidApp
class GeneticsApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // Initialization code for libraries like Timber, Crashlytics, etc. would go here.
    }
}
