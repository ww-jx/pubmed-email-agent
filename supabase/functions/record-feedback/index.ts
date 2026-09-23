import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { verifySignedParams } from '../_shared/signed-link.ts'

Deno.serve(async (req) => {
  const supabaseAdmin = createClient(
    Deno.env.get('SUPABASE_URL') || Deno.env.get('DB_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') || Deno.env.get('DB_SERVICE_ROLE_KEY')!
  )

  const baseUrl = Deno.env.get('FEEDBACK_BASE_URL')!;
  const successUrl = `${baseUrl.replace(/\/+$/, '')}/success`;
  const errorUrl = `${baseUrl.replace(/\/+$/, '')}/error`;
  const secret = Deno.env.get('LINK_SIGNING_SECRET')!;

  try {
    const url = new URL(req.url)

    const { user_id, pmid, rating } = await verifySignedParams(
      url, secret, ['user_id', 'pmid', 'rating']
    )

    const ratingValue = parseInt(rating, 10)
    if (!(ratingValue >= 1 && ratingValue <= 5)) {
      throw new Error("Invalid rating.")
    }

    const { error } = await supabaseAdmin
      .from('article_feedback')
      .upsert({ user_id, pmid, rating: ratingValue }, { onConflict: 'user_id,pmid' })

    if (error) {
      throw error
    }

    return Response.redirect(successUrl, 303)

  } catch (error) {
    console.error('Feedback recording error:', error.message)
    return Response.redirect(errorUrl, 303)
  }
})
