package com.animalgenetics.data.repository

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.animalgenetics.domain.model.Language
import com.animalgenetics.domain.repository.PreferencesRepository
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

@Singleton
class PreferencesRepositoryImpl @Inject constructor(
    @ApplicationContext private val context: Context
) : PreferencesRepository {

    private object PreferencesKeys {
        val LANGUAGE = stringPreferencesKey("language")
    }

    override suspend fun setPreferredLanguage(language: Language) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.LANGUAGE] = language.name
        }
    }

    override suspend fun getPreferredLanguage(): Language {
        val languageName = context.dataStore.data
            .map { preferences ->
                preferences[PreferencesKeys.LANGUAGE] ?: Language.ENGLISH.name
            }.first()
        return Language.valueOf(languageName)
    }
}
