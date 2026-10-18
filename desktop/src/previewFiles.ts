export async function verifiedPreviewBytes(paths: [string, string], hashes: [string, string], read: (path: string) => Promise<ArrayBuffer> = async path => {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`无法读取配对预览：${response.status}`);
  return response.arrayBuffer();
}): Promise<[ArrayBuffer, ArrayBuffer]> {
  const bytes = await Promise.all(paths.map(read));
  for (let i = 0; i < 2; i++) {
    if (!/^[0-9a-f]{64}$/.test(hashes[i])) throw new Error('配对预览缺少文件 SHA256');
    const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes[i]))].map(v => v.toString(16).padStart(2, '0')).join('');
    if (hash !== hashes[i]) throw new Error('配对预览在读取时已更新，等待同一请求的下一帧');
  }
  return bytes as [ArrayBuffer, ArrayBuffer];
}
