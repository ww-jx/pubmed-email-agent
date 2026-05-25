import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

Deno.serve(async (req) => {
  const supabaseAdmin = createClient(
    Deno.env.get('SUPABASE_URL') || Deno.env.get('DB_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') || Deno.env.get('DB_SERVICE_ROLE_KEY')!
  )

  const baseUrl = Deno.env.get('FEEDBACK_BASE_URL')!;
  const successUrl = `${baseUrl.replace(/\/+$/, '')}/success`;
  const errorUrl = `${baseUrl.replace(/\/+$/, '')}/error`;

  try {
    const url = new URL(req.url)
    const user_id = url.searchParams.get('user_id')
    const pmid = url.searchParams.get('pmid')
    const rating = parseInt(url.searchParams.get('rating') || '0', 10)

    if (!user_id || !pmid || !(rating >= 1 && rating <= 5)) {
      throw new Error("Invalid or missing parameters.")
    }

    const { error } = await supabaseAdmin
      .from('article_feedback')
      .insert({ user_id, pmid, rating })

    if (error) {
      throw error
    }

    return Response.redirect(successUrl, 303)

  } catch (error) {
    console.error('Feedback recording error:', error.message)
    return Response.redirect(errorUrl, 303)
  }
})
