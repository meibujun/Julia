package com.animalgenetics.core

import android.content.Context
import com.animalgenetics.domain.model.Language
import com.animalgenetics.domain.repository.PreferencesRepository
import dagger.hilt.android.qualifiers.ApplicationContext
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class LocalizationManager @Inject constructor(
    @ApplicationContext private val context: Context,
    private val preferencesRepository: PreferencesRepository
) {

    fun switchLanguage(language: Language) {
        // In a real app, this would trigger a configuration change
        // and activities would need to be recreated.
        // For now, we just save the preference.
        // GlobalScope.launch { preferencesRepository.setPreferredLanguage(language) }
    }

    private fun updateConfiguration(language: Language) {
        val locale = when (language) {
            Language.CHINESE_SIMPLIFIED -> Locale.SIMPLIFIED_CHINESE
            Language.ENGLISH -> Locale.ENGLISH
            Language.CHINESE_TRADITIONAL -> Locale.TRADITIONAL_CHINESE
        }

        val config = context.resources.configuration
        config.setLocale(locale)
        context.createConfigurationContext(config)
    }

    // Scientific term consistency across languages
    fun getScientificTerm(termKey: String, preferNative: Boolean = false): String {
        // val currentLanguage = runBlocking { preferencesRepository.getPreferredLanguage() }
        val currentLanguage = Language.ENGLISH // Placeholder

        return when (currentLanguage) {
            Language.CHINESE_SIMPLIFIED -> {
                if (preferNative) {
                    GeneticsTerminology.getChineseTerm(termKey)
                } else {
                    "${GeneticsTerminology.getChineseTerm(termKey)} (${GeneticsTerminology.getEnglishTerm(termKey)})"
                }
            }
            Language.ENGLISH -> GeneticsTerminology.getEnglishTerm(termKey)
            else -> GeneticsTerminology.getEnglishTerm(termKey)
        }
    }
}
