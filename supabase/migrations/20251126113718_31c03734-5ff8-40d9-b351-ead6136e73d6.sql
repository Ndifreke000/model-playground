-- Create analysis history table
CREATE TABLE IF NOT EXISTS public.analysis_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  prediction NUMERIC NOT NULL CHECK (prediction >= 0 AND prediction <= 1),
  confidence NUMERIC NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  is_authentic BOOLEAN NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.analysis_history ENABLE ROW LEVEL SECURITY;

-- Users can view their own analysis history
CREATE POLICY "Users can view their own analysis history"
ON public.analysis_history
FOR SELECT
TO authenticated
USING (auth.uid() = user_id);

-- Users can insert their own analysis
CREATE POLICY "Users can insert their own analysis"
ON public.analysis_history
FOR INSERT
TO authenticated
WITH CHECK (auth.uid() = user_id);

-- Users can delete their own analysis
CREATE POLICY "Users can delete their own analysis"
ON public.analysis_history
FOR DELETE
TO authenticated
USING (auth.uid() = user_id);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_analysis_history_user_id ON public.analysis_history(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_history_created_at ON public.analysis_history(created_at DESC);

-- Trigger to update updated_at
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_analysis_history_updated_at
BEFORE UPDATE ON public.analysis_history
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();