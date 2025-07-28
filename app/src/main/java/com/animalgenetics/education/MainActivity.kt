package com.animalgenetics.education

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.animalgenetics.feature.knowledgegraph.KnowledgeGraphActivity
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // For the walking skeleton, we directly launch our test activity.
        // The original logic will be restored later.
        startActivity(Intent(this, KnowledgeGraphActivity::class.java))
        finish()
    }
}
