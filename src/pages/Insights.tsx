import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Shield, ArrowLeft, TrendingUp, AlertTriangle, BarChart3 } from "lucide-react";
import { toast } from "sonner";
import { Progress } from "@/components/ui/progress";

interface Stats {
  total: number;
  authentic: number;
  misinformation: number;
  avgConfidence: number;
}

const Insights = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState<Stats>({
    total: 0,
    authentic: 0,
    misinformation: 0,
    avgConfidence: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuthAndLoadStats();
  }, []);

  const checkAuthAndLoadStats = async () => {
    const { data: { session } } = await supabase.auth.getSession();
    
    if (!session) {
      toast.error("Please sign in to view insights");
      navigate("/auth");
      return;
    }

    loadStats();
  };

  const loadStats = async () => {
    try {
      const { data, error } = await supabase
        .from("analysis_history")
        .select("is_authentic, confidence");

      if (error) throw error;

      const total = data?.length || 0;
      const authentic = data?.filter((a) => a.is_authentic).length || 0;
      const misinformation = total - authentic;
      const avgConfidence = total > 0
        ? data!.reduce((sum, a) => sum + a.confidence, 0) / total
        : 0;

      setStats({ total, authentic, misinformation, avgConfidence });
    } catch (error) {
      console.error("Error loading stats:", error);
      toast.error("Failed to load insights");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  const authenticPercent = stats.total > 0 ? (stats.authentic / stats.total) * 100 : 0;
  const misinfoPercent = stats.total > 0 ? (stats.misinformation / stats.total) * 100 : 0;

  return (
    <div className="min-h-screen bg-gradient-subtle">
      <div className="container mx-auto max-w-6xl px-4 py-8">
        <Button
          variant="ghost"
          onClick={() => navigate("/")}
          className="mb-6"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back
        </Button>

        <div className="flex items-center gap-3 mb-8">
          <BarChart3 className="w-8 h-8 text-primary" />
          <h1 className="text-3xl font-bold">Model Insights</h1>
        </div>

        {stats.total === 0 ? (
          <Card className="p-12 text-center">
            <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">No data yet</h3>
            <p className="text-muted-foreground mb-6">
              Analyze some news articles to see insights
            </p>
            <Button onClick={() => navigate("/")}>
              Analyze News
            </Button>
          </Card>
        ) : (
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <TrendingUp className="w-6 h-6 text-primary" />
                <h3 className="text-xl font-semibold">Total Analyses</h3>
              </div>
              <p className="text-4xl font-bold text-primary">{stats.total}</p>
              <p className="text-sm text-muted-foreground mt-2">
                Articles analyzed
              </p>
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <Shield className="w-6 h-6 text-success" />
                <h3 className="text-xl font-semibold">Average Confidence</h3>
              </div>
              <p className="text-4xl font-bold text-success">
                {Math.round(stats.avgConfidence * 100)}%
              </p>
              <p className="text-sm text-muted-foreground mt-2">
                Model confidence level
              </p>
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <Shield className="w-6 h-6 text-success" />
                <h3 className="text-xl font-semibold">Authentic News</h3>
              </div>
              <p className="text-4xl font-bold text-success mb-2">{stats.authentic}</p>
              <Progress value={authenticPercent} className="h-2 mb-2" />
              <p className="text-sm text-muted-foreground">
                {authenticPercent.toFixed(1)}% of total
              </p>
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <AlertTriangle className="w-6 h-6 text-destructive" />
                <h3 className="text-xl font-semibold">Misinformation</h3>
              </div>
              <p className="text-4xl font-bold text-destructive mb-2">{stats.misinformation}</p>
              <Progress value={misinfoPercent} className="h-2 mb-2" />
              <p className="text-sm text-muted-foreground">
                {misinfoPercent.toFixed(1)}% of total
              </p>
            </Card>

            <Card className="p-6 md:col-span-2">
              <h3 className="text-xl font-semibold mb-4">About This Model</h3>
              <div className="space-y-3 text-sm text-muted-foreground">
                <p>
                  <strong>Context:</strong> This AI model is specifically trained to understand Nigerian news patterns, including political, social, and cultural contexts.
                </p>
                <p>
                  <strong>Technology:</strong> Powered by advanced language models with real-time analysis capabilities.
                </p>
                <p>
                  <strong>Accuracy:</strong> The model provides confidence scores to help you assess the reliability of each analysis.
                </p>
                <p className="text-warning">
                  <strong>Important:</strong> While highly accurate, no automated system is perfect. Always verify important information through multiple trusted sources.
                </p>
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};

export default Insights;
