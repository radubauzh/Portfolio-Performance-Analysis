"""
Portfolio Performance Analyzer

Analyzes portfolio performance with:
- Position-level returns decomposed into local and FX components
- Sector allocation and performance tracking
- RAG-based narrative generation using news articles
- Support for GPT-5 reasoning models and standard chat models
"""
from xmlrpc import client
import pandas as pd
import os
import glob
from typing import Dict, List, Tuple
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

class PortfolioAnalyzer:
    # Target sector weights for the portfolio
    TARGET_WEIGHTS = {
        "Information Technology": 0.11,
        "Health Care": 0.11,
        "Financials": 0.14,
        "Consumer Discretionary": 0.18,
        "Industrials": 0.22,
        "Consumer Staples": 0.24,
    }
    
    def __init__(self, excel_path: str, news_dir: str = "data/news", db_path: str = "data/chroma_db"):
        load_dotenv()
        self.excel_path = excel_path 
        self.news_dir = news_dir
        self.db_path = db_path
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "MYMODEL")
        self.portfolio_df = self.load_portfolio_data()
        self.news_data = self.load_all_news()
        self.vector_db = self.setup_vector_db()

    def load_portfolio_data(self) -> pd.DataFrame:
        """Load and clean the provided portfolio data with proper GBp handling."""
        df = pd.read_excel(self.excel_path)

        # Normalize columns
        rename = {
            "security_des": "ticker", "name": "name", "crncy": "currency",
            "gics_sector_name": "sector", "Price0": "price0", "Price1": "price1",
            "Position0": "position0", "Position1": "position1",
            "ExchangeRateUSD0": "fx0", "ExchangeRateUSD1": "fx1"
        }
        df = df.rename(columns=rename)

        # Verify columns and ensure numeric
        if missing := [c for c in rename.values() if c not in df.columns]:
            raise ValueError(f"Missing required columns: {missing}")

        numeric = ["price0", "price1", "position0", "position1", "fx0", "fx1"]
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["ticker", "sector"] + numeric)
        
        # Calculate returns
        df["local_return"] = (df["price1"] - df["price0"]) / df["price0"]
        df["fx_return"] = (df["fx1"] - df["fx0"]) / df["fx0"]
        df["total_return_usd"] = (1 + df["local_return"]) * (1 + df["fx_return"]) - 1
        
        # Calculate position values
        df["start_value_usd"] = df["price0"] * df["position0"] * df["fx0"]
        df["end_value_usd_perf"] = df["price1"] * df["position0"] * df["fx1"]
        df["end_value_usd_actual"] = df["price1"] * df["position1"] * df["fx1"]
        
        # Calculate P&L and return
        df["pnl_usd"] = df["end_value_usd_perf"] - df["start_value_usd"]
        df["return_pct"] = df["pnl_usd"] / df["start_value_usd"].replace(0, pd.NA)

        return df.reset_index(drop=True)

    def calculate_performance(self) -> dict[str, object]:
        """Compute portfolio metrics with proper weighted return calculation."""
        df = self.portfolio_df.copy()

        # Portfolio totals
        total_start = float(df["start_value_usd"].sum())
        total_end_perf = float(df["end_value_usd_perf"].sum())
        total_pnl = total_end_perf - total_start
        
        df["weight"] = df["start_value_usd"] / total_start
        total_return = float((df["total_return_usd"] * df["weight"]).sum())

        # Identify top contributors and detractors by absolute P&L
        top_contributors = (
            df[df["pnl_usd"] > 0]
            .sort_values("pnl_usd", ascending=False)
            .head(3)
            .reset_index(drop=True)
        )
        top_detractors = (
            df[df["pnl_usd"] < 0]
            .sort_values("pnl_usd")
            .head(3)
            .reset_index(drop=True)
        )

        sector_data = []
        for sector in df["sector"].unique():
            sector_df = df[df["sector"] == sector]
            sector_start = float(sector_df["start_value_usd"].sum())
            sector_weight = sector_start / total_start
            sector_weights = sector_df["start_value_usd"] / sector_start
            sector_return = float((sector_df["total_return_usd"] * sector_weights).sum())
            sector_pnl = float(sector_df["pnl_usd"].sum())
            sector_end_value = float(sector_df["end_value_usd_perf"].sum())
            sector_end_weight = sector_end_value / total_end_perf if total_end_perf else 0.0
            weight_change = sector_end_weight - sector_weight

            sector_data.append({
                "sector": sector,
                "start_value_usd": sector_start,
                "end_value_usd": sector_end_value,
                "pnl_usd": sector_pnl,
                "sector_return_pct": sector_return,
                "portfolio_weight": sector_weight,
                "end_portfolio_weight": sector_end_weight,
                "weight_change_pct": weight_change
            })
        
        sectors = pd.DataFrame(sector_data).sort_values("pnl_usd", ascending=False)

        sectors['target_weights'] = sectors['sector'].map(self.TARGET_WEIGHTS).fillna(0)

        contrib_cols = ["ticker", "name", "currency", "sector", "start_value_usd", 
                       "pnl_usd", "local_return", "fx_return", "total_return_usd"]
        
        return {
            "portfolio_start_value_usd": total_start,
            "portfolio_end_value_usd": total_end_perf,
            "total_pnl_usd": total_pnl,
            "total_return_pct": total_return,
            "top_contributors": top_contributors[contrib_cols] if len(top_contributors) > 0 else pd.DataFrame(),
            "top_detractors": top_detractors[contrib_cols] if len(top_detractors) > 0 else pd.DataFrame(),
            "sector_breakdown": sectors,
            "all_positions": df[contrib_cols].sort_values("pnl_usd", ascending=False)
        }

    def load_all_news(self) -> Dict[str, pd.DataFrame]:
        """Load all news CSV files from data/news directory."""
        news_dict = {}
        csv_files = glob.glob(os.path.join(self.news_dir, "*_news_*.csv"))
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            ticker = filename.split("_news_")[0]
            df = pd.read_csv(file_path)
            news_dict[ticker] = df
            
        return news_dict

    def setup_vector_db(self) -> chromadb.Collection:
        """Create vector DB and embed all news articles using OpenAI. Persists to disk."""
        client = chromadb.PersistentClient(path=self.db_path)
        
        embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-large"
        )
        
        try:
            collection = client.get_collection(
                name="portfolio_news",
                embedding_function=embedding_fn
            )
            return collection
        except:
            print("Creating new embeddings...")
            collection = client.create_collection(
                name="portfolio_news",
                embedding_function=embedding_fn
            )
        
        documents = []
        metadatas = []
        ids = []
        
        idx = 0
        for ticker, news_df in self.news_data.items():
            for _, row in news_df.iterrows():
                text = f"{row['title']}. {row.get('summary', '')}"
                documents.append(text)
                
                metadatas.append({
                    "ticker": ticker,
                    "title": row["title"],
                    "date": str(row.get("date", "")),
                    "url": row.get("url", ""),
                    "source": row.get("source", ""),
                    "author": row.get("author", row.get("source", ""))
                })
                ids.append(f"news_{idx}")
                idx += 1
        
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Created {len(documents)} embeddings and saved to {self.db_path}")
        
        return collection

    def get_market_context(self, ticker: str, company_name: str, n_results: int = 3) -> List[Dict]:
        """Retrieve relevant news for a ticker using RAG."""
        query = f"{company_name} {ticker} corporate news announcements earnings"
        results = self.vector_db.query(
            query_texts=[query],
            n_results=n_results,
            where={"ticker": ticker}
        )
        
        context_list = []
        if results['metadatas'] and results['metadatas'][0]:
            for metadata, document in zip(results['metadatas'][0], results['documents'][0]):
                context_list.append({
                    "title": metadata["title"],
                    "date": metadata["date"],
                    "url": metadata["url"],
                    "source": metadata.get("source", ""),
                    "author": metadata.get("author", metadata.get("source", "")),
                    "text": document
                })
        
        return context_list

    def generate_narrative(self) -> str:
        """Generate and return formatted portfolio narrative with citations."""
        narrative_text, citations = self._generate_narrative_internal()
        return self._format_output(narrative_text, citations)

    def _generate_narrative_internal(self) -> Tuple[str, List[Dict]]:
        """Internal method: Generate portfolio narrative with LLM and proper citations."""
        perf = self.calculate_performance()
        
        # Prepare position data
        top_contrib = perf["top_contributors"]
        top_detract = perf["top_detractors"]
        
        # Get market context for key positions
        citations = []
        citation_idx = 1
        
        position_contexts = {}
        for idx, row in top_contrib.iterrows():
            news = self.get_market_context(row["ticker"], row["name"], n_results=3)
            position_contexts[row["ticker"]] = {
                "news": news,
                "data": row
            }
            for article in news:
                citations.append({
                    "idx": citation_idx,
                    "ticker": row["ticker"],
                    "title": article["title"],
                    "date": article["date"],
                    "url": article["url"],
                    "source": article["source"],
                    "author": article["author"]
                })
                citation_idx += 1
        
        for idx, row in top_detract.iterrows():
            news = self.get_market_context(row["ticker"], row["name"], n_results=3)
            position_contexts[row["ticker"]] = {
                "news": news,
                "data": row
            }
            for article in news:
                citations.append({
                    "idx": citation_idx,
                    "ticker": row["ticker"],
                    "title": article["title"],
                    "date": article["date"],
                    "url": article["url"],
                    "source": article["source"],
                    "author": article["author"]
                })
                citation_idx += 1
        
        # Build prompt for LLM
        prompt = self._build_narrative_prompt(perf, position_contexts, citations)
        
        # Call LLM
        narrative = self._call_llm(prompt)
        
        return narrative, citations

    def _build_narrative_prompt(self, perf: Dict, position_contexts: Dict, citations: List[Dict]) -> str:
        """Build concise prompt for narrative generation."""
        top_contrib = perf["top_contributors"]
        top_detract = perf["top_detractors"]
        sectors = perf["sector_breakdown"]
        
        # Build position summaries
        contrib_summary = ""
        for idx, row in top_contrib.iterrows():
            ticker = row["ticker"]
            news_items = position_contexts.get(ticker, {}).get("news", [])
            news_text = ""
            for article in news_items:
                cit_idx = [c['idx'] for c in citations if c['ticker'] == ticker and c['title'] == article['title']][0]
                news_text += f"[{cit_idx}] {article['text']}\n"
            
            contrib_summary += f"{row['name']} ({ticker}): +{row['total_return_usd']*100:.2f}%, ${row['pnl_usd']:,.0f} P&L, Local: {row['local_return']*100:.2f}%, FX: {row['fx_return']*100:.1f}%\nNews: {news_text}\n"
        
        detract_summary = ""
        for idx, row in top_detract.iterrows():
            ticker = row["ticker"]
            news_items = position_contexts.get(ticker, {}).get("news", [])
            news_text = ""
            for article in news_items:
                cit_idx = [c['idx'] for c in citations if c['ticker'] == ticker and c['title'] == article['title']][0]
                news_text += f"[{cit_idx}] {article['text']}\n"
            
            detract_summary += f"{row['name']} ({ticker}): {row['total_return_usd']*100:.2f}%, ${row['pnl_usd']:,.0f} P&L, Local: {row['local_return']*100:.2f}%, FX: {row['fx_return']*100:.1f}%\nNews: {news_text}\n"
        
        sector_summary = ", ".join([f"{s['sector']} ({s['sector_return_pct']*100:.1f}%)" for _, s in sectors.head(3).iterrows()])

        # Build sector weight change summary
        weight_lines = []
        for _, s in sectors.iterrows():
            sec = s["sector"]
            if sec in self.TARGET_WEIGHTS:
                start_w = s["portfolio_weight"] * 100
                end_w = s.get("end_portfolio_weight", 0) * 100
                change_pp = s.get("weight_change_pct", 0) * 100  # in percentage points
                tgt = self.TARGET_WEIGHTS[sec] * 100
                diff_vs_tgt = end_w - tgt
                status = "overweight" if diff_vs_tgt > 0.5 else ("underweight" if diff_vs_tgt < -0.5 else "near target")
                weight_lines.append(f"{sec}: {start_w:.1f}% -> {end_w:.1f}% (Δ {change_pp:+.1f}pp, target {tgt:.0f}% {status})")
        weights_summary = " | ".join(weight_lines)
        
        prompt = f"""Write a 6-10 sentence portfolio narrative for September 30 to October 24, 2025.

            Portfolio: {perf['total_return_pct']*100:.2f}% return, ${perf['total_pnl_usd']:,.0f} profit

            Top Contributors:
            {contrib_summary}

            Top Detractors:
            {detract_summary}

            Sectors: {sector_summary}
            Sector Weights: {weights_summary}

            Format:
            1. Opening: "The portfolio gained X.X% from September 30 to October 24, 2025, generating $XXX,XXX in profit."
            2. For each top contributor: "[Company] ([TICKER]) was the largest contributor, advancing X.X% and adding $XX,XXX to portfolio value following [specific event from news with citation]. [citation]" ALWAYS separate local vs FX when FX impact >5%.
            3. For each top detractor: "On the downside, [Company] ([TICKER]) was the primary drag, declining X.X% and reducing portfolio value by $XX,XXX. [Explain what happened based on news - e.g., 'Despite J.P. Morgan's maintained Buy rating and positive trial results [X][Y], the stock gained X.X% locally but unfavorable currency movements (-X.X%) resulted in a net decline.' OR if no relevant news: 'The stock declined X.X% with currency headwinds of -X.X% contributing to the total return.']. CRITICAL: ALWAYS explain FX vs local when FX impact >5%."
            4. Write FULL sentences for ALL top 3 contributors and detractors - never abbreviate or group them.
            5. Closing with sector/currency themes, AND explicitly mention notable sector weight shifts vs targets where material (>1pp change or meaningfully overweight/underweight).

            CITATION RULES (CRITICAL):
            - When you cite [X], you MUST mention WHAT that article says (e.g., "J.P. Morgan maintained Buy rating [X]" or "positive Phase III trial [Y]")
            - NEVER write "[X][Y][Z]" without explaining what those articles contain
            - If articles contain relevant information (analyst ratings, trials, earnings, deals), USE that information and cite it
            - If articles don't explain the price movement, omit citations and just state the return/P&L
            - Each of the top 3 contributors AND top 3 detractors deserves its own complete sentence with context"""
        
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API with support for both reasoning and chat completion models."""
        client = self.openai_client
        model = self.model

        # Use Responses API for GPT-5 reasoning models
        if model.startswith("gpt-5"):
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": "You are a professional portfolio analyst writing concise market narratives."},
                    {"role": "user", "content": prompt},
                ],
                reasoning={"effort": "low"},
                max_output_tokens=4000
            )
            text = getattr(resp, "output_text", "") or ""
            return text if text else str(resp)

        # Use Chat Completions API for standard models
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional portfolio analyst writing concise market narratives."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_completion_tokens=1500
        )
        return resp.choices[0].message.content or ""


    def _format_output(self, narrative: str, citations: List[Dict]) -> str:
        """Format final output with narrative and sources, and save to file."""
        output = narrative + "\n\nSources:\n"
        
        for cit in citations:
            author_source = cit['author'] if cit['author'] else cit['source']
            output += f"[{cit['idx']}] {cit['title']} - {author_source}, {cit['date']}\n"
            output += f"    {cit['url']}\n"
        
        os.makedirs("results", exist_ok=True)
        with open("results/portfolio_narrative.txt", "w") as f:
            f.write(output)
        
        return output