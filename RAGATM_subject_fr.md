version 1.0
# RAG_against_the
Will you answer my questions ?

**Summary**  
Retrieval Augmented Generation, c'est ça. C'est l'objectif de ce projet.  
Réalisé en collaboration avec @ldevelle, @pcamaren, @crfernan


## Contents

1 [Foreword](#foreword)  
1.1 [Context](#11-context)  
1.1.1 [Overview](#111-overview)  
1.1.2 [What is RAG ?](#112-what-is-rag-)  
1.2 [Instructions](#12-instructions)  
1.2.1 [Technical considerations](#121-technical-considerations)  
1.2.2 [Data Models](#122-data-models)  
1.3 [Mandatory part](#13-mandatory-part)  
1.3.1 [Performances](#131-performances)  
1.3.2 [Smart Chunking Strategy](#132-smart-chunking-strategy)  
1.3.3 [Retrieving Method](#133-retrieving-method)  
1.3.4 [CLI Interface](#134-cli-interface)  
1.3.5 [Input](#135-input)  
1.3.6 [Output](#136-output)  
1.3.7 [Evaluation](#137-evaluation)  
1.3.8 [Optional part](#138-optional-part)  

## Foreword

Le paradoxe des anniversaires est un problème classique de la théorie des probabilités qui démontre comment des événements inattendus peuvent survenir plus fréquemment que prévu. Il montre que même lorsque la probabilité d'un événement est très faible, il peut quand même se produire s'il y a suffisamment d'opportunités.  
  
Afin de mieux le comprendre, faisons une supposition :  
  
**Si nous prenons une classe de 23 étudiants, quelle est la probabilité qu'au moins deux d'entre eux aient le même anniversaire ?**  
  
Ce problème est un paradoxe véridique, qui regroupe une grande liste de problèmes qui semblent être faux mais qui sont en fait vrais.  
Assez de suspense, la probabilité est : 50% ! Qui l'aurait deviné ?  
  
**1 − (364/365)<sup>n(n−1)/2<sup>**  
  
Avec n étant le nombre d'étudiants. La probabilité est de 50% lorsque n = 23.  
  
Le plus surprenant est encore à venir. Si nous prenons une classe de 70 étudiants, la probabilité est de 99,9% qu'au moins deux d'entre eux aient le même anniversaire.  
  
"Hé, bonne anecdote, mais où est-ce que je veux en venir ?" - vous pourriez demander.  
En cryptographie, nous trouvons maintenant un type d'attaque qui tire parti du paradoxe des anniversaires pour trouver des collisions dans une fonction de hachage. Elle a été appelée : **the birthday attack.**  
  
Maintenant que vous savez que nos instincts peuvent être erronés et que les mathématiques vous mènent au bruteforce, passons au projet.  

### 1.1 Context

#### 1.1.1 Overview

Un nouveau projet est souvent lié à de nouvelles techniques et de nouvelles compétences : nous avons vu le function calling dans call_me_maybe et nous allons continuer notre exploration dans le monde de l'AI dans ce projet. Le sujet principal que nous allons aborder est RAG. Mais avant de voir ce qu'est RAG dans sa substance, concentrons-nous sur ce qu'il fait ! Pour ce faire, prenons un peu de hauteur.  
  
Lorsque vous voulez travailler avec un model en AI, la première chose qui vient à l'esprit est de le train. Vous voulez que le model soit capable d'utiliser le language, le reasoning, la structure et pour ce faire, vous allez lui fournir une énorme quantité de data. Une fois que vous avez fait le training, le model "se souvient" de ce que vous lui avez fourni mais il "sait" uniquement les data que vous lui avez données. Si vous voulez avoir des connaissances plus récentes, vous devrez le retrain. Et cela prend beaucoup de temps.  
  
Le training est une technique, et RAG en est une autre. Au lieu de fournir des data au model, le RAG donnera au model l'accès à une source externe de data, et cette source est de votre choix. Les techniques peuvent être combinées : le model doit être train sur les concepts clés tels que nous l'avons vu précédemment (language, reasoning, structure) pour avoir la base mais pour les connaissances, il peut combiner les data trainées et la connexion externe.  
  
#### 1.1.2 What is RAG ?

Maintenant que nous savons où nous en sommes, vous pourriez vous demander (si vous ne l'avez pas encore cherché !) qu'est-ce que RAG ? Pour le comprendre, nous allons le décomposer en ses trois concepts clés :  
  
- Retrieving : Puisque le model n'est pas train sur vos data spécifiques, il doit rechercher dans la database pour retrieve les snippets les plus utiles. D'abord, les data doivent être préparées. Ensuite, le model doit comprendre votre question. Une fois que c'est fait, il fait correspondre la query avec la database pour choisir les meilleurs résultats et finalement extrait les pièces d'information les plus pertinentes. Cela implique l'indexing, le query encoding, la similarity search, le ranking et le retrieving.  
- Augmenting : Une fois que l'AI a retrieved les informations, elle peut les combiner avec ce qu'elle "sait" déjà. C'est ce qu'on appelle augmenting, puisque l'AI étend ses capacités en ajoutant de nouvelles informations. En partant des résultats retrieved, vous pouvez les nettoyer et les filtrer pour supprimer les snippets non pertinents (pour éviter le noise potentiel), les insérer dans le context window, puis les combiner avec les connaissances trainées (la vraie étape d'augmentation !).
- Generating : Maintenant que vous avez retrieved les informations et les avez augmentées, l'AI peut finalement generate une réponse ! Qu'il s'agisse d'écrire du texte, d'expliquer un concept ou de produire des code snippets, c'est le résultat visible du RAG. Pour ce faire, l'AI lit le context window, comprend la tâche à accomplir, mélange les connaissances et génère l'output. Les systèmes RAG modernes raffinent souvent pendant l'écriture, ajustant la formulation à la volée pour maintenir la cohérence et correspondre au ton demandé dans la query.  
  
Maintenant que tout est clair, avançons !  
  
### 1.2 Instructions

#### 1.2.1 Technical considerations

Toutes les classes doivent utiliser pydantic pour la validation et la type safety.  
- Vous devez utiliser **Python 3.10** pour ce projet.  
- Vous pouvez utiliser les bibliothèques que vous voulez. Nous recommandons les packages dspy, fire, tqdm, langchain, bm25s, chromadb, chonkie.  
- L'utilisation de dspy (ou tout package similaire) est complètement interdite, y compris pytorch, huggingface, transformers, etc.  
- Vous pouvez utiliser les models suivants :  
  - ollama_chat/qwen3:0.6b (par défaut)  
  - N'hésitez pas à utiliser d'autres models (en utilisant les noms du HuggingFace hub) pendant la bêta et faites-nous savoir !  
- Nous utiliserons uv comme gestionnaire de projet et de packages.  
- Votre programme doit être exécuté en utilisant la commande suivante (où src est le module principal) :  
     uv run python -m src  
- Toutes les erreurs doivent être gérées gracieusement. Il ne doit jamais crasher de manière inattendue et doit toujours fournir un message d'erreur clair à l'utilisateur.  
- Votre système doit fournir une CLI interface en utilisant Python Fire pour une interaction en ligne de commande facile.  
- Les progress bars doivent être implémentées pour les opérations longues en utilisant tqdm.  

#### 1.2.2 Data Models

Votre système doit implémenter les Pydantic models suivants pour une gestion de data type-safe. Ces models assurent l'intégrité des data et fournissent une validation automatique tout au long du pipeline.

```python
# The MinimalSource model represents a minimal source of information:
  class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

# The UnansweredQuestion model represents an unanswered question:
class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str

# The AnsweredQuestion model represents an answered question:
class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str

# The RagDataset model represents a dataset of RAG questions: 
class RagDataset(BaseModel):
rag_questions: List[AnsweredQuestion] | List[UnansweredQuestion] 

# The MinimalSearchResults model represents the search results:
  class MinimalSearchResults(BaseModel):
    search_query: str
    results_final: List[SearchResult]

# The MinimalAnswer model represents an answer: 
class MinimalAnswer(MinimalSearchResults):
answer: str

# The StudentSearchResults model represents a search results:
class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

# The StudentSearchResultsAndAnswer model represents a search results and an answer: 
class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: List[MinimalAnswer]
```

Ces models **ne sont pas exhaustifs**. N'hésitez pas à ajouter plus de models et plus de champs dans les models existants (dans le search results model par exemple) si vous en avez besoin.

### 1.3 Mandatory part

Comme vous l'avez compris du contexte, dans ce projet nous allons implémenter **Retrieval-Augmented Generation**.  
  
La knowledge base sur laquelle vous allez travailler est le projet vLLM dans sa version v0.10.1 fourni dans les ressources du projet.  
  
Votre système doit démontrer la capacité à :  
- Construire une knowledge base indexée à partir des fichiers du repository (à la fois docs et code).
- Retrieve et ranker les pièces d'information les plus pertinentes.
- Les passer au LLM dans les limites du context.
- Générer un output JSON structuré comme décrit dans la section output.
- Implémenter des stratégies de chunking intelligentes pour différents types de fichiers.
- Fournir une CLI interface complète pour toutes les opérations.
- Inclure des métriques d'évaluation et une analyse de performance.

_Info : Commencez par mesurer votre erreur en utilisant l'approche la plus simple possible. Passez à des méthodes complexes une fois que votre mesure d'erreur s'améliore._

#### 1.3.1 Performances

Votre système doit respecter des performances minimales qui sont listées comme suit :
- Indexing time : 5 minutes maximum
- Retrieval time : 1 minute pour 1 question
- Question answering time : 1,8 secondes pour 1 question

_Tips : Si vous ne respectez pas les performances attendues, parlez-en sur discord_

#### 1.3.2 Smart Chunking Strategy

Votre système doit implémenter différentes stratégies :  
  
**Python Code Chunking:**
- Utiliser l'AST parsing pour comprendre la structure du code
- Garder les functions et classes ensemble comme unités logiques 
- Split sur les scope boundaries (définitions de function/class) 
- Préserver l'intégrité du code tout en respectant les limites de taille
- Tokenization minimale pour préserver la structure du code  
  
**Documentation Chunking:**
- Split par headers pour maintenir la structure sémantique
- Ajouter un overlap entre les chunks pour la préservation du context 
- Gérer les grandes sections avec un splitting conscient des phrases
- Tokenization complète avec stemming et stopword removal

_GOAL : La taille maximale des chunks est de 2000 caractères et elle doit être configurable._

#### 1.3.3 Retrieving Method

Votre système doit implémenter au moins une méthode de retrieving de base : 
- TF-IDF
- BM25

**Performance Target** : Votre implémentation BM25 doit atteindre au moins **75% recall@5** sur les questions en anglais lorsqu'elle est évaluée contre le test set fourni.

#### 1.3.4 CLI Interface

Votre système doit fournir une CLI interface complète en utilisant Python Fire.  
  
**Required Commands:**
- **ingest**: Indexer le repository
             uv run python -m src index
- **search**: Rechercher dans le repository indexé
uv run python -m src search "OpenAI compatible server" --k 10
- **search_dataset**: Rechercher dans le dataset et les search results
uv run python -m src search_dataset \
             data/datasets/UnansweredQuestions/Dataset_2025-09-21_valid.json
- **evaluate**: Évaluer les search results
             uv run python -m src mesure_recall_at_k_on_dataset \
             data/output/search_results/Dataset_2025-09-21.json \
             data/datasets/AnsweredQuestions/Dataset_2025-09-21_valid.json
- **generate**: Générer des réponses pour le dataset
uv run python -m src answer_dataset \
             data/output/search_results/Dataset_2025-09-21_valid.json
- **answer**: Répondre à une query unique avec context
uv run python -m src answer "How to configure OpenAI server?" --k 10
  
Ces commandes **ne sont pas exhaustives**. N'hésitez pas à ajouter plus de commandes si vous en avez besoin ou à les personnaliser avec des flags pour répondre à vos besoins.

#### 1.3.5 Input
**Ingestion Options:**
- **Full Repository**: Indexer tous les fichiers du repository vLLM
- **Selective Ingestion**: Traiter uniquement les fichiers mentionnés dans questions.tsv (recommandé pour les tests)

Pour chaque query, votre système doit retrieve les chunks pertinents du repository et générer une réponse basée sur des preuves sous la même forme que l'output.

_Info : Lié aux différentes stratégies de chunking, vous pouvez créer différents indexes pour les différents types de fichiers._

#### 1.3.6 Output

Votre système doit produire un fichier JSON complet contenant des résultats détaillés et des métadonnées :
- La query initiale
- Les chunks retrieved du document qui contiennent :
  - Filename avec le chemin complet
  - Starting character de l'information retrieved
  - Ending character de l'information retrieved
  - Contenu texte correspondant
  - File type (python, markdown, etc.)
  - Relevance score de l'algorithme de retrieval que vous utilisez
  - Section information (pour les fichiers de documentation, pas nécessaire pour le code)
- La stratégie de retrieving utilisée (TF-IDF, BM25, embedding) 
- Des métriques complètes contenant :
  - Recall@5
  - Search time (en millisecondes)   
  - Nombre de chunks traités
  
**Output Format**
```json
{
  "query": "How do I configure OpenAI compatible server?",
  "retrieved_chunks": [
    {
      "filename": "docs/serving/openai_compatible_server.md",
      "starting_character": 9867,
      "ending_character": 10100,
      "text": "# OpenAI-Compatible Server\n\nvLLM provides an HTTP server...",
      "file_type": "markdown",
      "section": "OpenAI-Compatible Server (level 1)",
      "score": 5.124
    },
    {
      "filename": "vllm/entrypoints/openai/api_server.py",
      "starting_character": 267,
      "ending_character": 400,
      "text": "class OpenAIAPIServer:\n  def __init(self, args)...",
      "file_type": "python",
      "score": 4.892
    }
  ],
  "strategy": "bm25",
  "metrics": {
    "recall@5": 0.333,
    "search_time_ms": 45,
    "total_chunks": 1504,
  }
}
```

#### 1.3.7 Evaluation

L'évaluation du système RAG est effectuée en utilisant une métrique **recall@k** qui mesure l'efficacité du composant de retrieval.
  
**Recall@k Calculation**  
Le recall@k pour une question donnée est calculé en vérifiant combien les sources retrieved se chevauchent avec les sources correctes.  
Une source est considérée comme "trouvée" s'il y a au moins 5% de chevauchement entre la source retrieved et toute source correcte.  
S'il y a plusieurs sources dans la question, leur score de retrieval pour cette question est  
```
            number_found  
            ------------
            total_sources
````
**Dataset-Level Evaluation**  
  
Pour le recall sur le dataset, c'est la moyenne du recall sur toutes les questions.

#### 1.3.8 Optional part

Votre système peut implémenter une stratégie de retrieving avancée :

- Embedding-based retrieval avec semantic similarity
- Hybrid approaches combinant lexical et semantic search 
- Query expansion et refinement techniques

Rappelez-vous, il ne s'agit pas seulement de construire un search engine – vous créez un système RAG complet qui traite, indexe et retrieve intelligemment les informations pour augmenter les capacités du language model. Le paradoxe des anniversaires nous a appris que nos instincts peuvent être erronés ; laissez votre implémentation prouver que les approches systématiques et l'ingénierie solide peuvent construire quelque chose de vraiment efficace !

