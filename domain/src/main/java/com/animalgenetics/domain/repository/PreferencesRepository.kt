package com.animalgenetics.domain.repository

import com.animalgenetics.domain.model.Language

interface PreferencesRepository {
    suspend fun setPreferredLanguage(language: Language)
    suspend fun getPreferredLanguage(): Language
}
