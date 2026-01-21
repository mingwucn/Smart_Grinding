import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from .xai.base_explainer import BaseExplainer
from .xai.gradcam import GradCAMExplainer
from .xai.integrated_gradients import IntegratedGradientsExplainer
from .xai.shap import SHAPExplainer

class XAIManager:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer
        
        # Convert layer name to module if needed
        if isinstance(target_layer, str):
            target_module = None
            for name, module in model.named_modules():
                if name == target_layer:
                    target_module = module
                    break
            if target_module is None:
                raise ValueError(f"Target layer '{target_layer}' not found in model")
            target_layer = target_module
        
        self.explainers = {
            "gradcam": GradCAMExplainer(model, target_layer),
            "integrated_gradients": IntegratedGradientsExplainer(model),
            "shap": SHAPExplainer(model)
        }
        # Placeholders for future explainers
        self.explainers["lime"] = None
        self.explainers["attention"] = None
    
    def run_analysis(self, input_dict, methods, sample_idx, input_type, report_path):
        results = {}
        
        for method in methods:
            if method not in self.explainers or self.explainers[method] is None:
                print(f"Method {method} not implemented yet")
                continue
                
            explainer = self.explainers[method]
            
            # Create method-specific directory
            method_dir = os.path.join(report_path, "xai_reports", method)
            os.makedirs(method_dir, exist_ok=True)
            
            # Run explanation
            explanation = explainer.explain(input_dict)
            
            # Generate visualization
            fig = explainer.visualize(explanation, input_dict)
            img_path = os.path.join(method_dir, f"{input_type}_sample{sample_idx}.png")
            fig.savefig(img_path)
            plt.close(fig)
            
            # Calculate metrics
            metrics = explainer.metrics(explanation)
            
            # Save metrics to JSON
            metrics_path = os.path.join(method_dir, f"{input_type}_sample{sample_idx}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Add to results
            results[method] = {
                "image": img_path,
                "metrics": metrics,
                "metrics_path": metrics_path
            }
        
        # Generate markdown report
        self.generate_markdown_report(results, sample_idx, input_type, report_path)
        
        return results
    
    def generate_markdown_report(self, results, sample_idx, input_type, report_path):
        # Create markdown content
        md_content = f"# XAI Report - Sample {sample_idx}\n\n"
        md_content += f"**Input Type:** {input_type}\n\n"
        
        md_content += "## Explanation Methods\n\n"
        
        # Table header
        md_content += "| Method | Visualization | Metrics |\n"
        md_content += "|--------|---------------|---------|\n"
        
        for method, data in results.items():
            img_rel_path = os.path.relpath(data["image"], report_path)
            metrics_rel_path = os.path.relpath(data["metrics_path"], report_path)
            
            md_content += (
                f"| {method.capitalize()} | "
                f"[![{method}]({img_rel_path})]({img_rel_path}) | "
                f"[View Metrics]({metrics_rel_path}) |\n"
            )
        
        md_content += "\n## Metrics Summary\n\n"
        
        # Create metrics table
        metrics_table = []
        for method, data in results.items():
            for metric, value in data["metrics"].items():
                metrics_table.append({
                    "Method": method,
                    "Metric": metric,
                    "Value": value
                })
        
        if metrics_table:
            df = pd.DataFrame(metrics_table)
            md_content += df.to_markdown(index=False)
        
        # Save markdown file
        md_path = os.path.join(report_path, "xai_reports", f"sample{sample_idx}_report.md")
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        return md_path
