import { useCallback, useEffect, useRef, useState } from 'react';
import { desktop, engineCall, subscribeEngine } from './engine';
import { DEFAULT_ROOT, catalogFrom, eventMatches } from './studioState';
import type { Catalog, Context } from './studioState';
import { readable, record } from './types';

const nativeTransport = { available: desktop, call: engineCall, subscribe: subscribeEngine };
export function useStudio(externalBusy = false, transport = nativeTransport) {
  const [catalog, setCatalog] = useState<Catalog>({ root: DEFAULT_ROOT, locations: [], scenes: [], captures: [], runs: [], external_imports: [] });
  const [context, setContext] = useState<Context>({});
  const [busy, setBusy] = useState<string | null>(null), busyRef = useRef(false);
  const [error, setError] = useState(''), [result, setResult] = useState<Record<string, unknown>>({});
  const [events, setEvents] = useState<Record<string, unknown>[]>([]);
  const contextRef = useRef(context); contextRef.current = context;
  // Let the worker return the configured data_root before pinning requests to
  // the catalog. A hard-coded first request would override relocated configs.
  const rootRef = useRef<string | undefined>(undefined);
  const report = useCallback((e: unknown) => setError(readable(e)), []);
  const read = useCallback(async (action: string, params: Record<string, unknown> = {}) => record(await transport.call('studio.operation', { root: rootRef.current, ...contextRef.current, action, ...params })), [transport]);
  const refresh = useCallback(async () => {
    if (!transport.available) return;
    try { const next = catalogFrom(await transport.call('studio.catalog', rootRef.current ? { root: rootRef.current } : {})); rootRef.current = next.root; setCatalog(next); } catch (e) { report(e); }
  }, [report, transport]);
  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    if (!transport.available) return;
    let stopped = false, dispose: (() => void) | undefined;
    void transport.subscribe(event => {
      const data = { ...event.data, ...record(event.data.detail) };
      if (!eventMatches(data, contextRef.current)) return;
      setEvents(previous => [...previous, { event: event.event, ...data, received: Date.now() }].slice(-500));
      if (data.output) setResult(previous => ({ ...previous, ...data }));
    }).then(fn => { if (stopped) fn(); else dispose = fn; }).catch(report);
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => { stopped = true; dispose?.(); window.clearInterval(timer); };
  }, [refresh, report, transport]);
  const operate = async (action: string, params: Record<string, unknown> = {}) => {
    if (busyRef.current || externalBusy) return null;
    busyRef.current = true; setBusy(action); setError('');
    try {
      const value = await read(action, params); setResult(value); await refresh(); return value;
    } catch (e) { report(e); await refresh(); return null; }
    finally { busyRef.current = false; setBusy(null); }
  };
  // Read/control use independent RPCs and remain available during a long operation.
  const control = async (params: Record<string, unknown>) => {
    try { const value = await read('control_training', params); setEvents(previous => [...previous, { event: 'control_requested', ...value }].slice(-500)); return value; }
    catch (e) { report(e); return null; }
  };
  return { catalog, context, setContext, busy, locked: !!busy || externalBusy || !transport.available, error, setError, result, events, report, read, refresh, operate, control };
}
export type StudioWork = ReturnType<typeof useStudio>;
