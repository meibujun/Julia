package com.animalgenetics.data.local

import androidx.room.Database
import androidx.room.RoomDatabase
import com.animalgenetics.data.local.dao.UserDao
import com.animalgenetics.data.local.entity.UserEntity

@Database(
    entities = [UserEntity::class],
    version = 1,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun userDao(): UserDao
}
