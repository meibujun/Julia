package com.animalgenetics.data.di

import android.content.Context
import androidx.room.Room
import com.animalgenetics.data.local.AppDatabase
import com.animalgenetics.data.local.dao.UserDao
import com.animalgenetics.data.remote.ChaoxingApi
import com.animalgenetics.data.repository.AchievementRepositoryImpl
import com.animalgenetics.data.repository.AuthenticationRepositoryImpl
import com.animalgenetics.data.repository.CaseRepositoryImpl
import com.animalgenetics.data.repository.EncryptionManager
import com.animalgenetics.data.repository.KnowledgeGraphRepositoryImpl
import com.animalgenetics.data.repository.PreferencesRepositoryImpl
import com.animalgenetics.data.repository.ProdEncryptionManager
import com.animalgenetics.data.repository.QuestionBankRepositoryImpl
import com.animalgenetics.data.repository.UserProgressRepositoryImpl
import com.animalgenetics.domain.repository.AchievementRepository
import com.animalgenetics.domain.repository.AuthenticationRepository
import com.animalgenetics.domain.repository.CaseRepository
import com.animalgenetics.domain.repository.KnowledgeGraphRepository
import com.animalgenetics.domain.repository.PreferencesRepository
import com.animalgenetics.domain.repository.QuestionBankRepository
import com.animalgenetics.domain.repository.UserProgressRepository
import com.google.firebase.auth.FirebaseAuth
import dagger.Binds
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {

    @Binds
    @Singleton
    abstract fun bindAuthenticationRepository(impl: AuthenticationRepositoryImpl): AuthenticationRepository

    @Binds
    @Singleton
    abstract fun bindKnowledgeGraphRepository(impl: KnowledgeGraphRepositoryImpl): KnowledgeGraphRepository

    @Binds
    @Singleton
    abstract fun bindUserProgressRepository(impl: UserProgressRepositoryImpl): UserProgressRepository

    @Binds
    @Singleton
    abstract fun bindQuestionBankRepository(impl: QuestionBankRepositoryImpl): QuestionBankRepository

    @Binds
    @Singleton
    abstract fun bindAchievementRepository(impl: AchievementRepositoryImpl): AchievementRepository

    @Binds
    @Singleton
    abstract fun bindCaseRepository(impl: CaseRepositoryImpl): CaseRepository

    @Binds
    @Singleton
    abstract fun bindPreferencesRepository(impl: PreferencesRepositoryImpl): PreferencesRepository

    @Binds
    @Singleton
    abstract fun bindEncryptionManager(impl: ProdEncryptionManager): EncryptionManager
}

@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient {
        return OkHttpClient.Builder()
            .addInterceptor(HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BODY
            })
            .build()
    }

    @Provides
    @Singleton
    fun provideRetrofit(okHttpClient: OkHttpClient): Retrofit {
        return Retrofit.Builder()
            .baseUrl("https://api.animalgenetics.com/") // Placeholder URL
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
    }

    @Provides
    @Singleton
    fun provideChaoxingApi(retrofit: Retrofit): ChaoxingApi {
        return retrofit.create(ChaoxingApi::class.java)
    }

    @Provides
    @Singleton
    fun provideFirebaseAuth(): FirebaseAuth {
        return FirebaseAuth.getInstance()
    }
}

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {

    @Provides
    @Singleton
    fun provideAppDatabase(@ApplicationContext context: Context): AppDatabase {
        return Room.databaseBuilder(
            context,
            AppDatabase::class.java,
            "animal_genetics.db"
        ).build()
    }

    @Provides
    @Singleton
    fun provideUserDao(appDatabase: AppDatabase): UserDao {
        return appDatabase.userDao()
    }
}
