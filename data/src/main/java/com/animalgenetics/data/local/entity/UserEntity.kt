package com.animalgenetics.data.local.entity

import androidx.room.Embedded
import androidx.room.Entity
import androidx.room.PrimaryKey
import com.animalgenetics.domain.model.User
import com.animalgenetics.domain.model.UserProfileData
import com.animalgenetics.domain.model.UserRole

@Entity(tableName = "users")
data class UserEntity(
    @PrimaryKey
    val id: String,
    val chaoxingId: String,
    val email: String,
    val displayName: String,
    val role: UserRole,
    val institution: String,
    @Embedded val profileData: UserProfileDataEntity
)

data class UserProfileDataEntity(
    val grade: String?,
    val major: String?,
    val studentId: String?
)

// Mapper functions to convert between domain model and entity
fun UserEntity.toDomain(): User = User(
    id = this.id,
    chaoxingId = this.chaoxingId,
    email = this.email,
    displayName = this.displayName,
    role = this.role,
    institution = this.institution,
    profileData = UserProfileData(
        grade = this.profileData.grade,
        major = this.profileData.major,
        studentId = this.profileData.studentId
    )
)

fun User.toEntity(): UserEntity = UserEntity(
    id = this.id,
    chaoxingId = this.chaoxingId,
    email = this.email,
    displayName = this.displayName,
    role = this.role,
    institution = this.institution,
    profileData = UserProfileDataEntity(
        grade = this.profileData.grade,
        major = this.profileData.major,
        studentId = this.profileData.studentId
    )
)
