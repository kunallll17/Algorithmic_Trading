"""
Report generation module using fpdf.
"""
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
import pandas as pd

class PDF(FPDF):
    def header(self):
        # Logo could go here
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Algorithmic Trading Project Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_plot(self, image_path, width=150):
        if os.path.exists(image_path):
            self.image(image_path, w=width, x=(210-width)/2)
            self.ln(10)
        else:
            print(f"Warning: Image not found at {image_path}")

class ReportGenerator:
    """Generates a PDF report for the project."""
    
    def __init__(self, output_path="results/final_report.pdf"):
        self.pdf = PDF()
        self.output_path = output_path
        self.pdf.add_page()
        
    def generate_report(self, backtest_results: pd.DataFrame, optimization_summary=None):
        """Builds and saves the report."""
        
        # 1. Introduction
        self.pdf.chapter_title("1. Project Overview")
        self.pdf.chapter_body(
            "This report summarizes the performance of various algorithmic trading strategies including "
            "Momentum, Mean Reversion, and Pairs Trading. It also includes results from Reinforcement "
            "Learning experiments."
        )
        
        # 2. Backtest Performance
        self.pdf.chapter_title("2. Strategy Performance Summary")
        self.pdf.chapter_body("The table below highlights the top performing strategies based on Sharpe Ratio.")
        
        # Add simpler text table for top strategies
        if not backtest_results.empty:
            top_strategies = backtest_results.drop_duplicates().nlargest(5, "sharpe_ratio")
            
            self.pdf.set_font('Courier', '', 10)
            header = f"{'Strategy':<20} {'Symbol':<10} {'Sharpe':<10} {'Return':<10}"
            self.pdf.cell(0, 10, header, 0, 1)
            
            for _, row in top_strategies.iterrows():
                line = f"{row['strategy']:<20} {row['symbol']:<10} {row['sharpe_ratio']:.2f}      {row['total_return']:.2%}"
                self.pdf.cell(0, 10, line, 0, 1)
            
            self.pdf.ln()

        # 3. Visualizations
        self.pdf.add_page()
        self.pdf.chapter_title("3. Performance Visualizations")
        
        self.pdf.chapter_body("Overall performance dashboard:")
        self.pdf.add_plot("results/performance_dashboard.png")
        
        self.pdf.chapter_body("Strategy comparison chart:")
        self.pdf.add_plot("results/strategy_comparison.png")
        
        # 4. Optimization
        self.pdf.add_page()
        self.pdf.chapter_title("4. Optimization Results")
        self.pdf.chapter_body("Optimization results for Momentum and Mean Reversion strategies.")
        
        self.pdf.add_plot("results/momentum_optimization.png")
        
        # 5. Reinforcement Learning
        self.pdf.add_page()
        self.pdf.chapter_title("5. Reinforcement Learning Experiment")
        self.pdf.chapter_body(
            "An RL agent was trained using Proximal Policy Optimization (PPO). "
            "The training process involved a custom gym environment simulating trading logic."
        )
        
        # Save
        self.pdf.output(self.output_path)
        print(f"Report saved to {self.output_path}")
