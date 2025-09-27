import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

Deno.serve(async (req) => {
  const supabaseAdmin = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
  )

  const url = new URL(req.url)
  const user_id = url.searchParams.get('user_id')
  const pmid = url.searchParams.get('pmid')
  const rating = parseInt(url.searchParams.get('rating') || '0', 10)

  if (user_id && pmid && rating >= 1 && rating <= 5) {
    await supabaseAdmin.from('article_feedback').insert({ user_id, pmid, rating })
  }

  return Response.redirect(Deno.env.get('REDIRECT_URL')!, 303)

})
