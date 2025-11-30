import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Multi-perspective analysis prompts
const analysisPrompts = {
  factual: `You are a FACT-CHECKER AI. Analyze this text for factual accuracy.
Focus on:
- Verifiable claims and statistics
- Named sources and citations
- Logical consistency
- Historical accuracy
- Scientific accuracy

Return JSON:
{
  "score": <0-1, 0=completely false, 1=verified facts>,
  "verdict": "<Verified|Partially Verified|Unverified|False>",
  "findings": ["<specific factual findings>"],
  "reasoning": "<step-by-step fact-checking reasoning>"
}`,

  linguistic: `You are a LINGUISTIC ANALYSIS AI. Analyze this text for language patterns.
Focus on:
- Sensationalist language and clickbait patterns
- Emotional manipulation tactics
- Grammatical quality and professionalism
- Hedging vs absolute statements
- Source attribution language

Return JSON:
{
  "score": <0-1, 0=highly manipulative, 1=neutral/professional>,
  "verdict": "<Professional|Neutral|Sensationalist|Manipulative>",
  "patterns": ["<identified language patterns>"],
  "reasoning": "<linguistic analysis reasoning>"
}`,

  sentiment: `You are a SENTIMENT & BIAS AI. Analyze this text for emotional bias.
Focus on:
- Political bias indicators
- Emotional loading of words
- One-sided vs balanced presentation
- Target audience manipulation
- Fear/anger/outrage triggering

Return JSON:
{
  "score": <0-1, 0=heavily biased, 1=balanced>,
  "verdict": "<Balanced|Slightly Biased|Biased|Heavily Biased>",
  "bias_direction": "<left|right|neutral|unclear>",
  "emotional_triggers": ["<identified emotional triggers>"],
  "reasoning": "<bias analysis reasoning>"
}`,

  source: `You are a SOURCE CREDIBILITY AI. Analyze this text for source reliability.
Focus on:
- Attribution to named sources
- Official vs anonymous sources
- Expert credentials mentioned
- Document/report citations
- Journalistic standards

Return JSON:
{
  "score": <0-1, 0=no credible sources, 1=well-sourced>,
  "verdict": "<Well-Sourced|Partially Sourced|Poorly Sourced|Unsourced>",
  "sources_identified": ["<sources mentioned>"],
  "credibility_issues": ["<issues found>"],
  "reasoning": "<source analysis reasoning>"
}`,

  propaganda: `You are a PROPAGANDA DETECTION AI. Analyze this text for propaganda techniques.
Focus on:
- Appeal to authority/emotion/fear
- Bandwagon effect
- Card stacking (selective facts)
- Name calling/labeling
- Transfer/testimonial techniques
- Repetition and slogans

Return JSON:
{
  "score": <0-1, 0=heavy propaganda, 1=no propaganda>,
  "verdict": "<Clean|Minor Techniques|Moderate Propaganda|Heavy Propaganda>",
  "techniques_found": ["<propaganda techniques identified>"],
  "reasoning": "<propaganda analysis reasoning>"
}`
};

async function runAnalysis(text: string, type: string, prompt: string, apiKey: string) {
  console.log(`Running ${type} analysis...`);
  
  const response = await fetch('https://ai.gateway.lovable.dev/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'google/gemini-2.5-flash',
      messages: [
        { role: 'system', content: prompt },
        { role: 'user', content: `Analyze this text:\n\n${text}` }
      ],
      response_format: { type: "json_object" }
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(`${type} analysis error:`, response.status, errorText);
    throw new Error(`${type} analysis failed: ${response.status}`);
  }

  const data = await response.json();
  return JSON.parse(data.choices[0].message.content);
}

async function generateSynthesis(text: string, analyses: any, apiKey: string) {
  console.log('Generating synthesis...');
  
  const synthesisPrompt = `You are an expert misinformation analyst. Given the following multi-perspective analysis results, provide a comprehensive synthesis.

FACTUAL ANALYSIS: ${JSON.stringify(analyses.factual)}
LINGUISTIC ANALYSIS: ${JSON.stringify(analyses.linguistic)}
SENTIMENT ANALYSIS: ${JSON.stringify(analyses.sentiment)}
SOURCE ANALYSIS: ${JSON.stringify(analyses.source)}
PROPAGANDA ANALYSIS: ${JSON.stringify(analyses.propaganda)}

Provide a final synthesis in JSON:
{
  "overall_score": <0-1 weighted average>,
  "overall_verdict": "<Authentic|Likely Authentic|Uncertain|Likely Misinformation|Misinformation>",
  "confidence": <0-1>,
  "executive_summary": "<2-3 sentence summary>",
  "key_concerns": ["<top concerns if any>"],
  "strengths": ["<credibility strengths if any>"],
  "recommendation": "<what the reader should do>",
  "detailed_reasoning": "<comprehensive chain-of-thought analysis explaining how you weighed each factor>"
}`;

  const response = await fetch('https://ai.gateway.lovable.dev/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'google/gemini-2.5-flash',
      messages: [
        { role: 'system', content: synthesisPrompt },
        { role: 'user', content: `Original text:\n\n${text}` }
      ],
      response_format: { type: "json_object" }
    }),
  });

  if (!response.ok) {
    throw new Error('Synthesis failed');
  }

  const data = await response.json();
  return JSON.parse(data.choices[0].message.content);
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { text, mode } = await req.json();

    if (!text || text.trim().length === 0) {
      return new Response(
        JSON.stringify({ error: 'Text is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');
    if (!LOVABLE_API_KEY) {
      console.error('LOVABLE_API_KEY is not configured');
      return new Response(
        JSON.stringify({ error: 'AI service not configured' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log('Starting advanced multi-algorithm analysis...');

    // Run all analyses in parallel for efficiency
    const [factual, linguistic, sentiment, source, propaganda] = await Promise.all([
      runAnalysis(text, 'factual', analysisPrompts.factual, LOVABLE_API_KEY),
      runAnalysis(text, 'linguistic', analysisPrompts.linguistic, LOVABLE_API_KEY),
      runAnalysis(text, 'sentiment', analysisPrompts.sentiment, LOVABLE_API_KEY),
      runAnalysis(text, 'source', analysisPrompts.source, LOVABLE_API_KEY),
      runAnalysis(text, 'propaganda', analysisPrompts.propaganda, LOVABLE_API_KEY),
    ]);

    const analyses = { factual, linguistic, sentiment, source, propaganda };
    
    // Generate synthesis
    const synthesis = await generateSynthesis(text, analyses, LOVABLE_API_KEY);

    console.log('Analysis complete');

    return new Response(
      JSON.stringify({
        analyses,
        synthesis,
        timestamp: new Date().toISOString()
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  } catch (error) {
    console.error('Error in analyze-news-advanced function:', error);
    
    if (error instanceof Error && error.message?.includes('429')) {
      return new Response(
        JSON.stringify({ error: 'Rate limit exceeded. Please try again later.' }),
        { status: 429, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }
    
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Analysis failed' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
