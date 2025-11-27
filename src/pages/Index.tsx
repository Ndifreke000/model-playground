import { TextAnalyzer } from "@/components/TextAnalyzer";
import { Shield, Brain, History, LogOut, Menu } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const Index = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => subscription.unsubscribe();
  }, []);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    toast.success("Signed out successfully");
  };

  return (
    <div className="min-h-screen bg-gradient-subtle">
      {/* Navigation */}
      <nav className="bg-card/80 backdrop-blur-md border-b border-border sticky top-0 z-50 shadow-sm">
        <div className="container mx-auto max-w-6xl px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 cursor-pointer" onClick={() => navigate("/")}>
              <Shield className="w-8 h-8 text-primary" />
              <div>
                <h2 className="text-xl font-bold">AI News Detector</h2>
                <p className="text-xs text-muted-foreground">Misinformation Analysis</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {user ? (
                <>
                  <Button
                    variant="ghost"
                    onClick={() => navigate("/history")}
                    className="hidden sm:flex items-center gap-2"
                  >
                    <History className="w-4 h-4" />
                    History
                  </Button>
                  <Button
                    variant="ghost"
                    onClick={() => navigate("/insights")}
                    className="hidden sm:flex items-center gap-2"
                  >
                    <Brain className="w-4 h-4" />
                    Insights
                  </Button>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="sm:hidden">
                        <Menu className="w-5 h-5" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => navigate("/history")}>
                        <History className="w-4 h-4 mr-2" />
                        History
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => navigate("/insights")}>
                        <Brain className="w-4 h-4 mr-2" />
                        Insights
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={handleSignOut}>
                        <LogOut className="w-4 h-4 mr-2" />
                        Sign Out
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                  <Button
                    variant="outline"
                    onClick={handleSignOut}
                    className="hidden sm:flex items-center gap-2"
                  >
                    <LogOut className="w-4 h-4" />
                    Sign Out
                  </Button>
                </>
              ) : (
                <Button onClick={() => navigate("/auth")} variant="default">
                  Sign In
                </Button>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="bg-gradient-hero py-16 px-4 shadow-lg">
        <div className="max-w-4xl mx-auto text-center">
          <div className="flex justify-center mb-6">
            <div className="p-4 bg-primary-foreground/10 rounded-2xl backdrop-blur-sm">
              <Shield className="w-16 h-16 text-primary-foreground" />
            </div>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-primary-foreground mb-4">
            AI News Detector
          </h1>
          <p className="text-lg md:text-xl text-primary-foreground/90 max-w-2xl mx-auto">
            Combat misinformation with AI-powered analysis using advanced language models
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
              <h3 className="font-semibold text-card-foreground">AI-Powered</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Advanced language models analyze news authenticity with high accuracy
            </p>
          </Card>

          <Card className="p-6 bg-card hover:shadow-lg transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Shield className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold text-card-foreground">Context-Aware</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Understands political, social, and cultural news patterns globally
            </p>
          </Card>

          <Card className="p-6 bg-card hover:shadow-lg transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <History className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold text-card-foreground">Track History</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Save and review your analysis history {user ? "automatically" : "(sign in required)"}
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
            AI-powered news analysis built with advanced language models.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
