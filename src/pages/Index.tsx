import { TextAnalyzer } from "@/components/TextAnalyzer";
import { Shield, Brain, Database } from "lucide-react";
import { Card } from "@/components/ui/card";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <header className="bg-gradient-hero py-16 px-4 shadow-lg">
        <div className="max-w-4xl mx-auto text-center">
          <div className="flex justify-center mb-6">
            <div className="p-4 bg-primary-foreground/10 rounded-2xl backdrop-blur-sm">
              <Shield className="w-16 h-16 text-primary-foreground" />
            </div>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-primary-foreground mb-4">
            Fake News Detector
          </h1>
          <p className="text-lg md:text-xl text-primary-foreground/90 max-w-2xl mx-auto">
            AI-powered misinformation detection using advanced neural networks and TF-IDF analysis
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 py-12">
        {/* Features */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <Card className="p-6 bg-card hover:shadow-lg transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Brain className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold text-card-foreground">Neural Network</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Powered by a trained PyTorch model with 64-neuron hidden layer
            </p>
          </Card>

          <Card className="p-6 bg-card hover:shadow-lg transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Database className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold text-card-foreground">79K Dataset</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Trained on extensive misinformation dataset from Kaggle
            </p>
          </Card>

          <Card className="p-6 bg-card hover:shadow-lg transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Shield className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold text-card-foreground">TF-IDF Features</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Advanced text vectorization for accurate classification
            </p>
          </Card>
        </div>

        {/* Analyzer */}
        <TextAnalyzer />
      </main>

      {/* Footer */}
      <footer className="mt-16 py-8 px-4 border-t border-border">
        <div className="max-w-4xl mx-auto text-center">
          <p className="text-sm text-muted-foreground">
            Built with PyTorch, TF-IDF, and React. Model architecture: Input → 64 neurons → Sigmoid output
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
