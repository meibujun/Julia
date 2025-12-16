package com.animalgenetics.feature.knowledgegraph

import android.os.Bundle
import android.widget.TextView
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

@AndroidEntryPoint
class KnowledgeGraphActivity : AppCompatActivity() {

    private val viewModel: KnowledgeGraphViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_knowledge_graph)

        val textView = findViewById<TextView>(R.id.concepts_text)

        lifecycleScope.launch {
            viewModel.uiState.collectLatest { state ->
                when (state) {
                    is KnowledgeGraphUiState.Loading -> {
                        textView.text = "Loading..."
                    }
                    is KnowledgeGraphUiState.Success -> {
                        textView.text = state.concepts.joinToString("\n") { it.name }
                    }
                    is KnowledgeGraphUiState.Error -> {
                        textView.text = state.message
                    }
                }
            }
        }

        // Trigger the data load
        viewModel.loadConcepts()
    }
}
