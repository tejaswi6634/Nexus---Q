"""Prompt complexity classification using a Scikit-Learn TF-IDF + LogisticRegression pipeline."""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from typing import Tuple
import numpy as np


class ComplexityClassifier:
    """Classifies incoming prompts as Simple, Medium, or Complex."""

    LABELS = ["Simple", "Medium", "Complex"]

    def __init__(self) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ])
        self._train_bootstrap()

    def _train_bootstrap(self) -> None:
        texts, labels = [], []

        simple = [
            "What is the capital of France?",
            "Hello how are you doing today?",
            "What time is it right now?",
            "Define the word photosynthesis",
            "Who invented the telephone?",
            "Translate hello to Spanish",
            "What color is the sky?",
            "How many days are in a week?",
            "What is two plus two?",
            "Name three programming languages",
            "What does HTML stand for?",
            "Who wrote Romeo and Juliet?",
            "What is the boiling point of water?",
            "Say hello in French",
            "When was Python created?",
            "What is an integer?",
            "List primary colors",
            "What is the square root of 16?",
            "What continent is Brazil in?",
            "What is the speed of light?",
        ]

        medium = [
            "Explain the difference between supervised and unsupervised learning",
            "Write a Python function to sort a list using merge sort algorithm",
            "Compare and contrast REST APIs with GraphQL endpoints",
            "Describe the Model View Controller architecture pattern",
            "How does garbage collection work in Java virtual machines?",
            "Explain the TCP three way handshake process in networking",
            "Write a SQL query to find duplicate records in a table",
            "What are the SOLID principles in software engineering?",
            "How does blockchain consensus work in proof of stake?",
            "Explain the CAP theorem with real world examples",
            "Implement binary search in Python with edge case handling",
            "How does HTTPS encryption protect data in transit?",
            "Describe the differences between Docker and virtual machines",
            "Write unit tests for a REST API endpoint",
            "Explain how a convolutional neural network processes images",
            "What are database normalization forms and why do they matter?",
            "How does load balancing work across multiple servers?",
            "Explain the observer design pattern with code example",
            "What is the difference between threads and processes?",
            "How do microservices communicate through message queues?",
        ]

        complex_ = [
            "Design a distributed system architecture for a real-time stock trading platform with fault tolerance including microservices communication patterns and database sharding strategies",
            "Implement a custom garbage collector for a programming language runtime with generational collection mark and sweep and concurrent collection phases",
            "Create a comprehensive security audit framework for a microservices architecture covering authentication authorization encryption and intrusion detection across all service boundaries",
            "Design a machine learning pipeline for real-time fraud detection that handles millions of transactions per second with feature engineering model serving and feedback loops",
            "Architect a multi-tenant SaaS platform with horizontal scaling tenant isolation custom domain support and automated infrastructure provisioning",
            "Build a compiler for a domain-specific language with type inference pattern matching algebraic data types and optimizing code generation",
            "Design a fault-tolerant distributed database with ACID guarantees multi-region replication automatic failover and conflict resolution strategies",
            "Create a real-time recommendation engine using collaborative filtering with online learning cold start handling and A/B testing infrastructure",
            "Implement a custom Kubernetes operator for database lifecycle management including automated backups point-in-time recovery and zero-downtime schema migrations",
            "Design an end-to-end MLOps pipeline with model versioning experiment tracking automated retraining canary deployments and drift detection",
            "Build a real-time data processing system using event sourcing and CQRS patterns with exactly-once delivery guarantees and temporal query support",
            "Design a comprehensive observability platform with distributed tracing log aggregation metrics collection anomaly detection and automated incident response",
            "Implement a custom consensus protocol for a distributed system that handles network partitions byzantine faults and provides linearizable consistency",
            "Create a privacy-preserving federated learning framework with differential privacy secure aggregation and model poisoning detection",
            "Design a global content delivery architecture with edge computing serverless functions intelligent caching and real-time content personalization",
            "Build a quantum-resistant cryptographic protocol suite for enterprise communications with forward secrecy key management and certificate transparency",
            "Implement a distributed transaction coordinator supporting saga pattern with compensating transactions timeout handling and dead letter queues",
            "Design an autonomous database tuning system using reinforcement learning for index selection query optimization and resource allocation",
            "Create a multi-modal AI inference serving platform with dynamic batching model sharding GPU memory management and request prioritization",
            "Architect a zero-trust security framework for hybrid cloud environments with continuous verification micro-segmentation and policy-as-code enforcement",
        ]

        for t in simple:
            texts.append(t)
            labels.append(0)
        for t in medium:
            texts.append(t)
            labels.append(1)
        for t in complex_:
            texts.append(t)
            labels.append(2)

        self.pipeline.fit(texts, labels)

    def classify(self, text: str) -> str:
        prediction = self.pipeline.predict([text])[0]
        return self.LABELS[prediction]

    def classify_with_confidence(self, text: str) -> Tuple[str, float]:
        prediction = self.pipeline.predict([text])[0]
        probabilities = self.pipeline.predict_proba([text])[0]
        return self.LABELS[prediction], float(probabilities[prediction])

    def complexity_index(self, text: str) -> int:
        """Return numeric index: Simple=0, Medium=1, Complex=2."""
        return int(self.pipeline.predict([text])[0])
