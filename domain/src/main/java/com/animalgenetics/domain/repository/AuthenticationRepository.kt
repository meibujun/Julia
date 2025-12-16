package com.animalgenetics.domain.repository

import com.animalgenetics.domain.model.User
import kotlinx.coroutines.flow.Flow

interface AuthenticationRepository {
    val currentUser: Flow<User?>
    suspend fun authenticateWithChaoxing(username: String, password: String): Result<User>
    suspend fun signOut()
}
