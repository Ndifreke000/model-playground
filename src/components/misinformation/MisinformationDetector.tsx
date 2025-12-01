/**
 * Misinformation Detector Component
 * UI for detecting fake news using the ML backend
 */
import { useState } from 'react';
import { misinformationAPI, PredictionResult, MisinformationAPIError } from '@/lib/api/misinformation.ts';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Loader2, AlertTriangle, CheckCircle2, Info } from 'lucide-react';

const EXAMPLE_TEXTS = [
    {
        label: 'Breaking News Example',
        text: 'Breaking: Major scientific breakthrough announced by leading research institute...'
    },
    {
        label: 'Political News Example',
        text: 'Government officials met today to discuss new policies affecting millions of citizens...'
    }
];

export function MisinformationDetector() {
    const [text, setText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!text.trim()) {
            setError('Please enter some text to analyze');
            return;
        }

        setIsLoading(true);
        setError(null);
        setResult(null);

        try {
            const prediction = await misinformationAPI.predict(text);
            setResult(prediction);
        } catch (err) {
            if (err instanceof MisinformationAPIError) {
                setError(err.message);
            } else {
                setError('An unexpected error occurred. Please try again.');
            }
        } finally {
            setIsLoading(false);
        }
    };

    const loadExample = (exampleText: string) => {
        setText(exampleText);
        setResult(null);
        setError(null);
    };

    const reset = () => {
        setText('');
        setResult(null);
        setError(null);
    };

    const getResultColor = (prediction: string) => {
        return prediction === 'fake' ? 'destructive' : 'default';
    };

    const getResultIcon = (prediction: string) => {
        return prediction === 'fake' ? <AlertTriangle className="h-5 w-5" /> : <CheckCircle2 className="h-5 w-5" />;
    };

    return (
        <div className="container mx-auto max-w-4xl p-6 space-y-6">
            <div className="space-y-2">
                <h1 className="text-3xl font-bold">Misinformation Detector</h1>
                <p className="text-muted-foreground">
                    Analyze news articles and text to detect potential misinformation using machine learning.
                </p>
            </div>

            <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription>
                    This tool uses a machine learning model trained on news articles. The model has ~67.5% accuracy
                    and should be used as a preliminary screening tool, not a definitive fact-checker.
                </AlertDescription>
            </Alert>

            <Card>
                <CardHeader>
                    <CardTitle>Enter Text to Analyze</CardTitle>
                    <CardDescription>
                        Paste a news article, headline, or any text you want to check for misinformation.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div className="space-y-2">
                            <Textarea
                                value={text}
                                onChange={(e) => setText(e.target.value)}
                                placeholder="Enter or paste text here... (minimum 10 characters)"
                                className="min-h-[200px] resize-y"
                                disabled={isLoading}
                            />
                            <div className="flex justify-between items-center text-sm text-muted-foreground">
                                <span>{text.length} / 10,000 characters</span>
                                {text.length > 0 && (
                                    <Button
                                        type="button"
                                        variant="ghost"
                                        size="sm"
                                        onClick={reset}
                                        disabled={isLoading}
                                    >
                                        Clear
                                    </Button>
                                )}
                            </div>
                        </div>

                        <div className="flex gap-2 flex-wrap">
                            <Button
                                type="submit"
                                disabled={isLoading || text.trim().length < 10}
                                className="flex-1 sm:flex-none"
                            >
                                {isLoading ? (
                                    <>
                                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                        Analyzing...
                                    </>
                                ) : (
                                    'Analyze Text'
                                )}
                            </Button>

                            {EXAMPLE_TEXTS.map((example) => (
                                <Button
                                    key={example.label}
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => loadExample(example.text)}
                                    disabled={isLoading}
                                >
                                    {example.label}
                                </Button>
                            ))}
                        </div>
                    </form>

                    {error && (
                        <Alert variant="destructive">
                            <AlertTriangle className="h-4 w-4" />
                            <AlertDescription>{error}</AlertDescription>
                        </Alert>
                    )}

                    {result && (
                        <Card className="border-2">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    {getResultIcon(result.prediction)}
                                    Analysis Result
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium">Prediction:</span>
                                    <Badge variant={getResultColor(result.prediction)} className="text-lg px-4 py-1">
                                        {result.prediction.toUpperCase()}
                                    </Badge>
                                </div>

                                <div className="space-y-2">
                                    <div className="flex justify-between text-sm">
                                        <span className="font-medium">Confidence:</span>
                                        <span className="font-mono">{(result.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="w-full bg-secondary rounded-full h-2">
                                        <div
                                            className={`h-2 rounded-full transition-all ${result.prediction === 'fake' ? 'bg-destructive' : 'bg-primary'
                                                }`}
                                            style={{ width: `${result.confidence * 100}%` }}
                                        />
                                    </div>
                                </div>

                                <Alert>
                                    <Info className="h-4 w-4" />
                                    <AlertDescription className="text-sm">
                                        {result.prediction === 'fake' ? (
                                            <>
                                                This text shows characteristics of potential misinformation.
                                                Please verify with additional fact-checking sources.
                                            </>
                                        ) : (
                                            <>
                                                This text appears to be legitimate news.
                                                However, always cross-reference important information with multiple sources.
                                            </>
                                        )}
                                    </AlertDescription>
                                </Alert>
                            </CardContent>
                        </Card>
                    )}
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle className="text-lg">How it Works</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm text-muted-foreground">
                    <p>
                        This detector uses a machine learning model trained on thousands of news articles.
                        The model analyzes text patterns, writing style, and language characteristics to
                        identify potential misinformation.
                    </p>
                    <p className="font-medium text-foreground">
                        Note: This is a screening tool with ~67.5% accuracy. Always verify important
                        information through multiple trusted sources and professional fact-checkers.
                    </p>
                </CardContent>
            </Card>
        </div>
    );
}
