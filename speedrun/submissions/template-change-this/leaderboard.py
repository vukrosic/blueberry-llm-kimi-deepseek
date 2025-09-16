#!/usr/bin/env python3
"""
T4 Speedrun Leaderboard

This module manages the leaderboard for the T4 speedrun challenge,
tracking best validation losses and performance metrics.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse


class SpeedrunLeaderboard:
    """Leaderboard for T4 speedrun challenge."""
    
    def __init__(self, leaderboard_file: str = "speedrun_leaderboard.json"):
        self.leaderboard_file = Path(__file__).parent / "results" / leaderboard_file
        self.leaderboard_file.parent.mkdir(exist_ok=True)
        self.entries = self.load_leaderboard()
    
    def load_leaderboard(self) -> List[Dict[str, Any]]:
        """Load leaderboard from file."""
        if self.leaderboard_file.exists():
            with open(self.leaderboard_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_leaderboard(self):
        """Save leaderboard to file."""
        with open(self.leaderboard_file, 'w') as f:
            json.dump(self.entries, f, indent=2)
    
    def add_entry(self, results: Dict[str, Any], participant_name: str = None):
        """Add a new entry to the leaderboard."""
        if not results.get('completed', False):
            print("❌ Cannot add incomplete speedrun to leaderboard")
            return False
        
        entry = {
            'timestamp': results['timestamp'],
            'participant': participant_name or "Anonymous",
            'final_val_loss': results['final_val_loss'],
            'final_val_accuracy': results.get('final_val_accuracy', 0.0),
            'final_val_perplexity': results.get('final_val_perplexity', 0.0),
            'total_time_minutes': results['total_time_minutes'],
            'final_step': results['final_step'],
            'steps_per_second': results['steps_per_second'],
            'memory_usage_gb': results.get('memory_usage_gb', 0.0),
            'config': results.get('config', {}),
            'time_exceeded': results.get('time_exceeded', False),
        }
        
        self.entries.append(entry)
        self.sort_leaderboard()
        self.save_leaderboard()
        
        print(f"✅ Added entry to leaderboard: Val Loss = {entry['final_val_loss']:.6f}")
        return True
    
    def sort_leaderboard(self):
        """Sort leaderboard by validation loss (ascending)."""
        self.entries.sort(key=lambda x: x['final_val_loss'])
    
    def get_top_entries(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N entries from leaderboard."""
        return self.entries[:n]
    
    def get_rank(self, val_loss: float) -> int:
        """Get rank for a given validation loss."""
        rank = 1
        for entry in self.entries:
            if entry['final_val_loss'] < val_loss:
                rank += 1
        return rank
    
    def print_leaderboard(self, top_n: int = 10):
        """Print formatted leaderboard."""
        print("\n" + "="*80)
        print("🏆 T4 SPEEDRUN LEADERBOARD")
        print("="*80)
        
        if not self.entries:
            print("📭 No entries yet. Be the first to complete a speedrun!")
            return
        
        top_entries = self.get_top_entries(top_n)
        
        print(f"{'Rank':<4} {'Participant':<15} {'Val Loss':<12} {'Time':<8} {'Steps':<8} {'Memory':<8}")
        print("-" * 80)
        
        for i, entry in enumerate(top_entries, 1):
            time_str = f"{entry['total_time_minutes']:.1f}m"
            steps_str = f"{entry['final_step']:,}"
            memory_str = f"{entry['memory_usage_gb']:.1f}GB"
            
            # Add warning for time exceeded
            time_warning = " ⚠️" if entry['time_exceeded'] else ""
            
            print(f"{i:<4} {entry['participant']:<15} {entry['final_val_loss']:<12.6f} "
                  f"{time_str:<8} {steps_str:<8} {memory_str:<8}{time_warning}")
        
        print("-" * 80)
        print(f"📊 Total entries: {len(self.entries)}")
        
        # Show best performance metrics
        if self.entries:
            best_entry = self.entries[0]
            print(f"\n🥇 Best Performance:")
            print(f"   Validation Loss: {best_entry['final_val_loss']:.6f}")
            print(f"   Accuracy: {best_entry['final_val_accuracy']:.4f}")
            print(f"   Perplexity: {best_entry['final_val_perplexity']:.2f}")
            print(f"   Time: {best_entry['total_time_minutes']:.2f} minutes")
            print(f"   Steps: {best_entry['final_step']:,}")
            print(f"   Speed: {best_entry['steps_per_second']:.2f} steps/sec")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get leaderboard statistics."""
        if not self.entries:
            return {}
        
        val_losses = [entry['final_val_loss'] for entry in self.entries]
        times = [entry['total_time_minutes'] for entry in self.entries]
        steps = [entry['final_step'] for entry in self.entries]
        
        return {
            'total_entries': len(self.entries),
            'best_val_loss': min(val_losses),
            'worst_val_loss': max(val_losses),
            'avg_val_loss': sum(val_losses) / len(val_losses),
            'best_time': min(times),
            'worst_time': max(times),
            'avg_time': sum(times) / len(times),
            'max_steps': max(steps),
            'min_steps': min(steps),
            'avg_steps': sum(steps) / len(steps),
        }
    
    def print_statistics(self):
        """Print leaderboard statistics."""
        stats = self.get_statistics()
        if not stats:
            print("📭 No statistics available - no entries yet")
            return
        
        print("\n" + "="*60)
        print("📊 LEADERBOARD STATISTICS")
        print("="*60)
        
        print(f"Total Entries: {stats['total_entries']}")
        print(f"\nValidation Loss:")
        print(f"  Best:  {stats['best_val_loss']:.6f}")
        print(f"  Worst: {stats['worst_val_loss']:.6f}")
        print(f"  Avg:   {stats['avg_val_loss']:.6f}")
        
        print(f"\nTime:")
        print(f"  Best:  {stats['best_time']:.2f} minutes")
        print(f"  Worst: {stats['worst_time']:.2f} minutes")
        print(f"  Avg:   {stats['avg_time']:.2f} minutes")
        
        print(f"\nSteps:")
        print(f"  Max: {stats['max_steps']:,}")
        print(f"  Min: {stats['min_steps']:,}")
        print(f"  Avg: {stats['avg_steps']:,}")
    
    def export_csv(self, filename: str = None):
        """Export leaderboard to CSV."""
        if not self.entries:
            print("📭 No entries to export")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"speedrun_leaderboard_{timestamp}.csv"
        
        csv_path = Path(__file__).parent / "results" / filename
        
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'participant', 'final_val_loss', 'final_val_accuracy',
                'final_val_perplexity', 'total_time_minutes', 'final_step',
                'steps_per_second', 'memory_usage_gb', 'time_exceeded'
            ])
            writer.writeheader()
            writer.writerows(self.entries)
        
        print(f"📊 Leaderboard exported to {csv_path}")
    
    def clear_leaderboard(self):
        """Clear all entries from leaderboard."""
        self.entries = []
        self.save_leaderboard()
        print("🗑️ Leaderboard cleared")


def load_results_from_file(results_file: str) -> Dict[str, Any]:
    """Load results from a JSON file."""
    results_path = Path(__file__).parent / "results" / results_file
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)


def main():
    """Main entry point for leaderboard management."""
    parser = argparse.ArgumentParser(description="T4 Speedrun Leaderboard")
    parser.add_argument("--show", action="store_true", help="Show leaderboard")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--add", type=str, help="Add results file to leaderboard")
    parser.add_argument("--participant", type=str, help="Participant name for new entry")
    parser.add_argument("--export", type=str, help="Export leaderboard to CSV file")
    parser.add_argument("--clear", action="store_true", help="Clear leaderboard")
    parser.add_argument("--top", type=int, default=10, help="Number of top entries to show")
    
    args = parser.parse_args()
    
    leaderboard = SpeedrunLeaderboard()
    
    if args.clear:
        leaderboard.clear_leaderboard()
    elif args.add:
        results = load_results_from_file(args.add)
        if results:
            leaderboard.add_entry(results, args.participant)
    elif args.export:
        leaderboard.export_csv(args.export)
    elif args.stats:
        leaderboard.print_statistics()
    else:
        leaderboard.print_leaderboard(args.top)


if __name__ == "__main__":
    main()
