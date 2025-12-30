"""
Quick Transfer Script
Run this to set up your complete project structure
"""

from pathlib import Path


# File manifest - All files you need to copy
FILE_MANIFEST = {
    'Core System Files': [
        'backtesting_engine.py',
        'feature_engineering.py',
        'ml_training_pipeline.py',
        'strategy_examples.py',
        'trading_dashboard_main.py',
        'oanda_integration.py',
        'sample_data_generator.py',
        'complete_workflow.py',
        'run_examples.py',
        'setup.py',
    ],
    'Monitoring & Recovery': [
        'model_failure_recovery.py',
        'monitoring_integration.py',
    ],
    'Configuration': [
        'config_template.py',
    ],
    'Documentation': [
        'README.md',
        'TRANSFER_GUIDE.md',
        'QUICK_START.md',
        'PROJECT_CONTEXT.md',
    ]
}


def create_project_structure(base_path='trading_system'):
    """
    Create complete directory structure
    """
    base = Path(base_path)

    directories = [
        'core',
        'strategies',
        'integrations',
        'tools',
        'data/historical',
        'data/models',
        'data/results',
        'tests',
        'docs/guides',
        'docs/api',
        'docs/examples',
        'notebooks',
        'logs',
    ]

    print(f"Creating project structure in: {base.absolute()}")

    for directory in directories:
        dir_path = base / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {directory}/")

    # Create __init__.py files
    init_dirs = ['core', 'strategies', 'integrations', 'tools', 'tests']
    for init_dir in init_dirs:
        init_file = base / init_dir / '__init__.py'
        init_file.touch()
        print(f"  ✓ Created: {init_dir}/__init__.py")

    print("\n✅ Directory structure complete!")
    return base


def create_file_templates(base_path):
    """
    Create template/placeholder files
    """
    base = Path(base_path)

    # .gitignore
    gitignore = base / '.gitignore'
    gitignore.write_text('''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Data
data/historical/*.csv
data/models/*.pkl
data/results/*.json
*.db

# Secrets
config.py
.env
*.key

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
logs/*.log
''')
    print("  ✓ Created: .gitignore")

    # README.md
    readme = base / 'README.md'
    readme.write_text('''# Forex Trading System with ML

Professional trading system with backtesting, machine learning, and live monitoring.

## Quick Start

1. **Setup**
   ```bash
   python setup.py
   cp config_template.py config.py
   # Edit config.py with your Oanda credentials
   ```

2. **Run Example**
   ```bash
   python tools/run_examples.py --example 1
   ```

3. **View Dashboard**
   ```bash
   streamlit run integrations/trading_dashboard_main.py
   ```

## Documentation

See `docs/` folder for complete documentation.

## Chat History

Original development chat: [Your Claude chat URL here]

## Status

- Core System: ✅ Complete
- ML Pipeline: ✅ Complete
- Dashboard: ✅ Complete
- Documentation: ✅ Complete

## Next Steps

1. Configure Oanda API credentials
2. Run backtests
3. Train ML models
4. Paper trade for 90 days
5. Deploy to live (carefully!)
''')
    print("  ✓ Created: README.md")

    # requirements.txt
    requirements = base / 'requirements.txt'
    requirements.write_text('''# Core Dependencies
pandas>=2.0.0
numpy>=1.24.0
python-dateutil>=2.8.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0

# Data Visualization
plotly>=5.18.0
streamlit>=1.30.0

# API Integration
oandapyV20>=0.7.2

# Utilities
requests>=2.31.0
tqdm>=4.66.0
''')
    print("  ✓ Created: requirements.txt")

    # VS Code workspace
    workspace = base / 'trading_system.code-workspace'
    workspace.write_text('''{
  "folders": [
    {
      "path": "."
    }
  ],
  "settings": {
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.rulers": [88]
  }
}
''')
    print("  ✓ Created: trading_system.code-workspace")

    print("\n✅ Template files created!")


def create_file_checklist(base_path):
    """
    Create a checklist of files to copy manually
    """
    base = Path(base_path)

    checklist = base / 'FILE_CHECKLIST.md'

    content = '''# File Transfer Checklist

## Instructions
For each file below:
1. Open the artifact in Claude chat
2. Click "Copy" button
3. Create the file in the destination folder
4. Paste the content
5. Check the box: [ ] → [x]

---

'''

    # Add files organized by category
    for category, files in FILE_MANIFEST.items():
        content += f'## {category}\n\n'
        for file in files:
            # Determine destination folder
            if file in ['backtesting_engine.py', 'feature_engineering.py',
                        'ml_training_pipeline.py', 'model_failure_recovery.py']:
                dest = f'core/{file}'
            elif file == 'strategy_examples.py':
                dest = f'strategies/{file}'
            elif file in ['oanda_integration.py', 'trading_dashboard_main.py']:
                dest = f'integrations/{file}'
            elif file in ['complete_workflow.py', 'run_examples.py', 'setup.py',
                          'sample_data_generator.py', 'monitoring_integration.py']:
                dest = f'tools/{file}'
            else:
                dest = file

            content += f'- [ ] `{dest}`\n'

        content += '\n'

    # Add documentation files
    content += '''## Documentation Files (Copy from chat to docs/)

- [ ] `docs/Complete_System_Integration_Guide.md`
- [ ] `docs/Recovery_Strategies_Guide.md`
- [ ] `docs/Quick_Reference_Card.md`
- [ ] `docs/Dependency_Tree.md`
- [ ] `docs/Codebase_Summary.md`

---

## After Copying All Files

Run these commands to verify:

```bash
# Activate virtual environment
source venv/bin/activate  # Mac/Linux
# OR
venv\\Scripts\\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Test imports
python -c "from core.backtesting_engine import BacktestEngine; print('✅ Import OK')"

# Run quick test
python tools/run_examples.py --example 1
```

## Total Files: {total}
- Core System: 10 files
- Monitoring: 2 files
- Configuration: 1 file
- Documentation: 5+ files

---

**Estimated time: 20-30 minutes**
'''

    total_files = sum(len(files) for files in FILE_MANIFEST.values())
    content = content.format(total=total_files)

    checklist.write_text(content)
    print("  ✓ Created: FILE_CHECKLIST.md")
    print(f"\n📋 Checklist created with {total_files} files to copy")

    return checklist


def create_quick_commands(base_path):
    """
    Create quick command reference
    """
    base = Path(base_path)

    commands = base / 'QUICK_COMMANDS.md'
    commands.write_text('''# Quick Commands Reference

## Setup (First Time)

```bash
# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate  # Mac/Linux
venv\\Scripts\\activate   # Windows

# Install
pip install -r requirements.txt

# Configure
cp config_template.py config.py
# Edit config.py with your Oanda credentials
```

## Daily Use

```bash
# Run quick backtest example
python tools/run_examples.py --example 1

# Run specific example (1-6)
python tools/run_examples.py --example 4

# Launch dashboard
streamlit run integrations/trading_dashboard_main.py

# Run full pipeline
python tools/complete_workflow.py --mode full
```

## Development

```bash
# Run tests
pytest tests/

# Generate sample data
python tools/sample_data_generator.py

# Check system health
python tools/monitoring_integration.py --check
```

## Claude Code Commands

```bash
# Interactive chat
claude-code chat

# Review code
claude-code review core/backtesting_engine.py

# Generate tests
claude-code test core/backtesting_engine.py

# Get help
claude-code help
```

## Git Commands

```bash
# Initialize repo
git init
git add .
git commit -m "Initial commit - Complete trading system"

# Daily workflow
git add .
git commit -m "Added feature X"
git push
```

## Debugging

```bash
# Python debugger
python -m pdb tools/run_examples.py

# Check imports
python -c "import sys; print(sys.path)"

# Verbose output
python -v tools/run_examples.py --example 1
```
''')
    print("  ✓ Created: QUICK_COMMANDS.md")


def main():
    """
    Main setup function
    """
    print("\n" + "=" * 60)
    print("TRADING SYSTEM - PROJECT SETUP")
    print("=" * 60 + "\n")

    # Get project location
    default_path = 'trading_system'
    path = input(f"Project directory [{default_path}]: ").strip() or default_path

    print(f"\nSetting up project in: {Path(path).absolute()}\n")

    # Create structure
    base_path = create_project_structure(path)

    print("\n" + "=" * 60)
    print("CREATING TEMPLATE FILES")
    print("=" * 60 + "\n")

    # Create templates
    create_file_templates(base_path)

    print("\n" + "=" * 60)
    print("CREATING CHECKLISTS")
    print("=" * 60 + "\n")

    # Create checklist
    _checklist_path = create_file_checklist(base_path)  # noqa: F841

    # Create quick commands
    create_quick_commands(base_path)

    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)

    print('''
Next Steps:

1. **Copy Files** (20-30 minutes)
   Open: {checklist_path}
   Follow the checklist to copy all artifacts from chat

2. **Create Virtual Environment**
   cd {path}
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3. **Configure**
   cp config_template.py config.py
   # Edit config.py with your Oanda credentials

4. **Test**
   python tools/run_examples.py --example 1

5. **Start Development**
   code .  # Open in VS Code

See QUICK_COMMANDS.md for all available commands.
''')


if __name__ == '__main__':
    main()
