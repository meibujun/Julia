package com.animalgenetics.feature.knowledgegraph

import android.annotation.SuppressLint
import android.content.Context
import android.util.AttributeSet
import android.util.Log
import android.webkit.JavascriptInterface
import android.webkit.WebView
import android.webkit.WebViewClient
import com.animalgenetics.domain.model.ConceptRelationship
import com.animalgenetics.domain.model.GeneticsConcept
import com.animalgenetics.domain.model.GraphData
import com.animalgenetics.domain.model.GraphEdge
import com.animalgenetics.domain.model.GraphNode
import com.google.gson.Gson

class KnowledgeGraphView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : WebView(context, attrs, defStyleAttr) {

    var onNodeClickListener: ((String) -> Unit)? = null
    private var isGraphReady = false
    private val pendingUpdates = mutableListOf<Pair<List<GeneticsConcept>, List<ConceptRelationship>>>()

    init {
        setupWebView()
        loadGraphVisualization()
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            loadWithOverviewMode = true
            useWideViewPort = true
        }

        addJavascriptInterface(GraphJsBridge(), "AndroidBridge")
        webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                // The onGraphReady call from JS is more reliable
            }
        }
    }

    private fun loadGraphVisualization() {
        loadUrl("file:///android_asset/knowledge_graph.html")
    }

    fun updateGraphData(concepts: List<GeneticsConcept>, relationships: List<ConceptRelationship>) {
        if (!isGraphReady) {
            pendingUpdates.add(concepts to relationships)
            return
        }

        val graphData = GraphData(
            nodes = concepts.map { concept ->
                GraphNode(
                    id = concept.id,
                    label = concept.name,
                    type = concept.type.name,
                    mastery = concept.userMastery,
                    properties = mapOf(
                        "difficulty" to concept.difficulty,
                        "prerequisites" to concept.prerequisites
                    )
                )
            },
            edges = relationships.map { rel ->
                GraphEdge(
                    source = rel.fromId,
                    target = rel.toId,
                    type = rel.type.name,
                    weight = rel.strength
                )
            }
        )

        val json = Gson().toJson(graphData)
        post {
            evaluateJavascript("javascript:updateGraph($json)") { result ->
                Log.d("KnowledgeGraph", "Graph updated: $result")
            }
        }
    }

    inner class GraphJsBridge {
        @JavascriptInterface
        fun onNodeClicked(nodeId: String) {
            post {
                onNodeClickListener?.invoke(nodeId)
            }
        }

        @JavascriptInterface
        fun onGraphReady() {
            post {
                isGraphReady = true
                pendingUpdates.forEach { update -> updateGraphData(update.first, update.second) }
                pendingUpdates.clear()
            }
        }
    }
}
