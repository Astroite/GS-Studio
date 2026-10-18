import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const desktop=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const manifest=JSON.parse(fs.readFileSync(path.join(desktop,'package.json'),'utf8'));
const seen=new Set(),parts=['GS Studio desktop — frontend dependency notices\n\nThis distribution uses the following open-source libraries. The engine/runtime/tool notices are supplied separately by runtime.json package metadata. No original commercial Postshot binaries, models or assets are included.\n'];
function visit(name,base=desktop){
  let current=base,dir;
  while(true){const candidate=path.join(current,'node_modules',name);if(fs.existsSync(path.join(candidate,'package.json'))){dir=candidate;break;}const parent=path.dirname(current);if(parent===current)throw Error(`Missing dependency ${name}`);current=parent;}
  if(seen.has(dir))return;seen.add(dir);
  const pkg=JSON.parse(fs.readFileSync(path.join(dir,'package.json'),'utf8'));
  const licenses=fs.readdirSync(dir).filter(v=>/^(license|licence|copying|notice)(\.|$)/i.test(v)&&fs.statSync(path.join(dir,v)).isFile());
  parts.push(`\n========== ${pkg.name} ${pkg.version} (${pkg.license??'see upstream license'}) ==========\n`);
  for(const file of licenses)parts.push(`\n${file}\n${fs.readFileSync(path.join(dir,file),'utf8')}\n`);
  for(const dependency of Object.keys(pkg.dependencies??{}))visit(dependency,dir);
}
for(const name of Object.keys(manifest.dependencies))visit(name);
fs.writeFileSync(path.join(desktop,'frontend-notices.txt'),parts.join('').trimEnd()+'\n','utf8');
process.stdout.write(`Frontend notices generated for ${seen.size} runtime packages.\n`);
