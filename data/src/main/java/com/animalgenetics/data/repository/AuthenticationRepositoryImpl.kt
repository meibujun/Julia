package com.animalgenetics.data.repository

import com.animalgenetics.data.local.dao.UserDao
import com.animalgenetics.data.local.entity.toDomain
import com.animalgenetics.data.local.entity.toEntity
import com.animalgenetics.data.remote.ChaoxingApi
import com.animalgenetics.domain.model.ChaoxingAuthRequest
import com.animalgenetics.domain.model.User
import com.animalgenetics.domain.model.UserRole
import com.animalgenetics.domain.repository.AuthenticationRepository
import com.google.firebase.auth.FirebaseAuth
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.tasks.await
import javax.inject.Inject
import javax.inject.Singleton

// Placeholder for a real encryption manager
interface EncryptionManager {
    fun encrypt(password: String): String
}

class ProdEncryptionManager @Inject constructor() : EncryptionManager {
    override fun encrypt(password: String): String {
        // In a real app, this would use a secure encryption algorithm.
        // For this placeholder, we'll just reverse the string as a mock "encryption".
        return password.reversed()
    }
}


@Singleton
class AuthenticationRepositoryImpl @Inject constructor(
    private val chaoxingApi: ChaoxingApi,
    private val firebaseAuth: FirebaseAuth,
    private val userDao: UserDao,
    private val encryptionManager: EncryptionManager
) : AuthenticationRepository {

    override val currentUser: Flow<User?> = userDao.getUserById(firebaseAuth.currentUser?.uid ?: "").map { it?.toDomain() }

    override suspend fun authenticateWithChaoxing(username: String, password: String): Result<User> {
        return try {
            // First, authenticate with the Chaoxing platform via our backend
            val chaoxingToken = chaoxingApi.authenticate(
                ChaoxingAuthRequest(username, encryptionManager.encrypt(password))
            )

            // Then, use the custom token from our backend to sign in to Firebase
            val firebaseUser = firebaseAuth.signInWithCustomToken(chaoxingToken.firebaseToken).await().user
                ?: throw IllegalStateException("Firebase user not found after authentication.")

            // Determine user role from Chaoxing data
            val userRole = when (chaoxingToken.userType.lowercase()) {
                "student" -> UserRole.STUDENT
                "teacher" -> UserRole.TEACHER
                else -> throw IllegalArgumentException("Unknown user type: ${chaoxingToken.userType}")
            }

            // Create domain user object from the token data
            val user = User(
                id = firebaseUser.uid,
                chaoxingId = chaoxingToken.userId,
                email = chaoxingToken.email,
                displayName = chaoxingToken.displayName,
                role = userRole,
                institution = chaoxingToken.institution,
                profileData = com.animalgenetics.domain.model.UserProfileData(
                    grade = chaoxingToken.grade,
                    major = chaoxingToken.major,
                    studentId = chaoxingToken.studentId
                )
            )

            // Save the authenticated user to the local database
            userDao.insertUser(user.toEntity())

            Result.success(user)
        } catch (e: Exception) {
            // Log the exception in a real app
            e.printStackTrace()
            Result.failure(e)
        }
    }

    override suspend fun signOut() {
        firebaseAuth.signOut()
        userDao.clearAll()
    }
}
