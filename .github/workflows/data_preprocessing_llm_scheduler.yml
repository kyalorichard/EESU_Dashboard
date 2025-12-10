name: Run Repression Extraction

# Run every 2 minutes
on:
  schedule:
    - cron: '*/2 * * * *'  # every 2 minutes
  workflow_dispatch:        # allows manual trigger

jobs:
  extract:
    runs-on: ubuntu-latest

    steps:
      # Checkout repository
      - name: Checkout code
        uses: actions/checkout@v4

      # Set up Python
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      # Install dependencies
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pandas openai python-dotenv langdetect tiktoken pyarrow

      # Run the script
      - name: Run extraction script
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          NOTIFY_EMAIL: ${{ secrets.NOTIFY_EMAIL }}
          SMTP_USER: ${{ secrets.SMTP_USER }}
          SMTP_PASS: ${{ secrets.SMTP_PASS }}
          SMTP_HOST: ${{ secrets.SMTP_HOST }}
          SMTP_PORT: ${{ secrets.SMTP_PORT }}
        run: |
          python data_preprocessing.py
