import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

Deno.serve(async (req) => {
  const supabaseAdmin = createClient(
    Deno.env.get('SUPABASE_URL') || Deno.env.get('DB_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') || Deno.env.get('DB_SERVICE_ROLE_KEY')!
  );

  const baseUrl = Deno.env.get('UNSUBSCRIBE_BASE_URL')!;
  const successUrl = `${baseUrl.replace(/\/+$/, '')}/success`;
  const errorUrl = `${baseUrl.replace(/\/+$/, '')}/error`;

  try {
    const url = new URL(req.url);
    const userId = url.searchParams.get('user_id');

    if (!userId) {
      throw new Error("User ID is required.");
    }

    const { error } = await supabaseAdmin
      .from('users')
      .update({ subscribed: false })
      .eq('id', userId);

    if (error) throw error;

    return Response.redirect(successUrl, 303);

  } catch (error) {
    console.error('Unsubscribe error:', error.message);
    return Response.redirect(errorUrl, 303);
  }
});
