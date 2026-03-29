"""ML-based prompt priority classification using TF-IDF + LogisticRegression.

Automatically infers the business priority (LOW / MEDIUM / HIGH / VIP) of an
incoming prompt so that routing, SLA enforcement, and burst-shedding decisions
do not require manual annotation.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from typing import Tuple
import numpy as np

from adaptive_ai_control_plane.ingestion.prompt_request import Priority


class PriorityClassifier:
    """Classifies incoming prompts into Priority.LOW / MEDIUM / HIGH / VIP."""

    LABELS: list[Priority] = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.VIP]

    def __init__(self) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=600, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ])
        self._train_bootstrap()

    def _train_bootstrap(self) -> None:
        texts: list[str] = []
        labels: list[int] = []

        low = [
            "What is the capital of France?",
            "Tell me a creative story about space exploration and alien civilizations",
            "Write a poem about artificial intelligence and the future of humanity",
            "What is machine learning?",
            "Write a haiku about programming",
            "Summarize the history of the Internet in three sentences",
            "What is recursion?",
            "Quick health-check ping",
            "Status report request",
            "Simple lookup query for user data",
            "Check database connection status",
            "Retrieve cached configuration values",
            "Log aggregation summary for today",
            "Quick metric snapshot request",
            "Heartbeat verification ping",
            "System uptime query",
            "Cache invalidation request",
            "Ping the load balancer",
            "Fetch API version information",
            "Quick DNS lookup request",
            "Health endpoint response check",
            "Service discovery query",
            "How does a hash table handle collisions?",
            "Write a children's bedtime story about a robot learning to paint",
            "What are the three laws of thermodynamics?",
            "What color is the sky?",
            "How many days are in a week?",
            "What is two plus two?",
            "Translate hello to Spanish",
            "Who wrote Romeo and Juliet?",
        ]

        medium = [
            "Write a Python function for binary search with edge-case handling",
            "Explain quantum computing basics and qubit entanglement",
            "How do neural networks learn from data?",
            "Explain the CAP theorem and its implications for distributed databases",
            "Write creative marketing copy for a new AI product launch campaign",
            "Explain how the TCP three-way handshake works in network protocols",
            "Write a creative screenplay outline for a science-fiction thriller",
            "Create an interactive data-visualization dashboard for real-time analytics",
            "Compare and contrast REST APIs with GraphQL endpoints",
            "Write a SQL query to find duplicate records in a table",
            "How does garbage collection work in Java virtual machines?",
            "What are the SOLID principles in software engineering?",
            "Describe the differences between Docker and virtual machines",
            "Explain how a convolutional neural network processes images",
            "What are database normalization forms and why do they matter?",
            "How does load balancing work across multiple servers?",
            "Explain the observer design pattern with code example",
            "Write unit tests for a REST API endpoint",
            "Implement binary search in Python with edge case handling",
            "How do microservices communicate through message queues?",
        ]

        high = [
            "Design a microservices architecture for an e-commerce platform with event sourcing and CQRS",
            "Optimize this SQL query for better performance across partitioned tables",
            "Implement a distributed consensus algorithm with Byzantine fault tolerance",
            "Build a real-time recommendation engine with collaborative filtering",
            "Design an end-to-end MLOps pipeline with model versioning and canary deployments",
            "Implement a custom memory allocator with garbage collection for a language runtime",
            "Architect a zero-trust security framework for hybrid-cloud environments",
            "Design a privacy-preserving federated learning system with differential privacy",
            "Build a quantum-resistant encryption protocol for enterprise communications",
            "Implement a distributed tracing system for microservices observability",
            "Design a multi-modal AI inference serving platform with dynamic batching and GPU management",
            "Design a distributed system architecture for a real-time stock trading platform with fault tolerance",
            "Architect a multi-tenant SaaS platform with horizontal scaling and tenant isolation",
            "Build a compiler for a domain-specific language with type inference and pattern matching",
            "Design a fault-tolerant distributed database with ACID guarantees and multi-region replication",
            "Create a real-time data processing system using event sourcing and CQRS patterns",
            "Design a comprehensive observability platform with distributed tracing and anomaly detection",
            "Implement a custom consensus protocol for a distributed system that handles network partitions",
            "Design an autonomous database tuning system using reinforcement learning for index selection",
            "Implement a custom Kubernetes operator for database lifecycle management",
        ]

        vip = [
            "Draft a GDPR compliance review for our data-pipeline architecture",
            "Review this contract clause for legal compliance with international trade law",
            "Create a comprehensive security audit framework for a multi-cloud Kubernetes deployment",
            "Draft an intellectual property licensing agreement for our open-source project",
            "What are the current CCPA regulations for handling consumer personal data?",
            "Analyze legal implications of deploying AI in healthcare diagnostics",
            "Conduct a regulatory compliance assessment for our financial trading platform",
            "Review HIPAA compliance requirements for patient data processing pipeline",
            "Draft a data processing agreement under GDPR Article 28 provisions",
            "Perform a SOX compliance audit of our financial reporting infrastructure",
            "Assess PCI DSS compliance for our payment processing microservices",
            "Legal review of cross-border data transfer mechanisms under Schrems II",
            "Draft a privacy impact assessment for our new recommendation engine",
            "Review export control regulations for our encryption software distribution",
            "Create a compliance framework for AI model governance and explainability",
            "Audit data retention policies against FERPA requirements for student records",
            "Legal analysis of liability issues in autonomous vehicle decision systems",
            "Draft regulatory filing for our new fintech lending product under TILA",
            "Review anti-money laundering compliance for cryptocurrency exchange integration",
            "Assess NIST cybersecurity framework alignment for our critical infrastructure systems",
        ]

        for t in low:
            texts.append(t)
            labels.append(0)
        for t in medium:
            texts.append(t)
            labels.append(1)
        for t in high:
            texts.append(t)
            labels.append(2)
        for t in vip:
            texts.append(t)
            labels.append(3)

        self.pipeline.fit(texts, labels)

    def classify(self, text: str) -> Priority:
        prediction = int(self.pipeline.predict([text])[0])
        return self.LABELS[prediction]

    def classify_with_confidence(self, text: str) -> Tuple[Priority, float]:
        prediction = int(self.pipeline.predict([text])[0])
        probabilities = self.pipeline.predict_proba([text])[0]
        return self.LABELS[prediction], float(probabilities[prediction])
