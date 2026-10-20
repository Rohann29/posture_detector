import { NextResponse } from 'next/server';
import { kv } from '@vercel/kv';

export async function POST(request) {
    try {
        const { status } = await request.json();
        await kv.set('posture_status', status);
        return NextResponse.json({ success: true, saved: status });
    } catch (error) {
        return NextResponse.json({ error: 'Failed to save' }, { status: 500 });
    }
}

export async function GET() {
    try {
        const status = await kv.get('posture_status') || 'good';
        return NextResponse.json({ status });
    } catch (error) {
        return NextResponse.json({ error: 'Failed to fetch' }, { status: 500 });
    }
}