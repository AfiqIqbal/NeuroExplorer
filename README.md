# NeuroExplorer

A context-aware file explorer that understands the content of different file types through modular file agents, featuring a modern, interactive UI with semantic search capabilities.

## ✨ Features

- **Modern, Responsive UI**
  - Clean, futuristic design with glassmorphism effects
  - Interactive search with real-time feedback
  - Smooth animations and transitions
  - Dark theme for comfortable extended use

- **Smart Search**
  - Semantic search across file contents and metadata
  - Real-time search suggestions
  - Filter by file type and agent
  - Visual feedback during search operations

- **File Analysis**
  - Specialized agents for different file types (text, image, audio, code)
  - Automatic content summarization
  - Tag generation and metadata extraction
  - Embedding generation for semantic search

- **Extensible Architecture**
  - Modular design for adding new file type support
  - Plugin system for custom agents
  - RESTful API for integration with other tools

## Project Structure

```
neuroexplorer/
├── src/
│   ├── agents/           # File type agents
│   ├── core/             # Core functionality
│   ├── database/         # Database models and connections
│   ├── ui/               # User interface components
│   └── utils/            # Utility functions
├── tests/                # Unit and integration tests
└── main.py               # Application entry point
```

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/AfiqIqbal/NeuroExplorer.git
   cd NeuroExplorer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python -m uvicorn src.ui.server:app --reload --port 8000
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## 🖥️ UI Features

### Search Interface
- **Interactive Search Bar**: Type to search with real-time feedback
- **Results Panel**: Expandable results that appear directly below the search bar
- **Filters**: Quickly narrow down results by file type or processing agent
- **Visual Feedback**: Animated sphere responds to your typing and search activity

### Keyboard Shortcuts
- `Enter`: Perform search
- `Escape`: Clear search
- `Tab`: Navigate between form elements
- `↑/↓`: Navigate search results (coming soon)

## 🛠️ Development

### Project Structure
```
neuroexplorer/
├── src/
│   ├── agents/           # File type agents
│   ├── core/             # Core functionality
│   ├── database/         # Database models and connections
│   ├── ui/               # User interface components
│   │   ├── static/       # Frontend assets (JS, CSS)
│   │   └── templates/    # HTML templates
│   └── utils/            # Utility functions
├── tests/                # Unit and integration tests
└── main.py               # Application entry point
```

## Adding New File Type Agents

1. Create a new agent class in `src/agents/`
2. Implement the `BaseAgent` interface
3. Register the agent in the agent registry
4. Support incremental indexing by accepting the optional `metadata` argument in `analyze()` and returning it with any cached values (e.g., `summary`, `tags`, `embedding`, `content_hash`).

## License

MIT
