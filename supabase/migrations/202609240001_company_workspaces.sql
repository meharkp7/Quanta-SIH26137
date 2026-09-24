-- Run once in the Supabase SQL editor. The browser never needs a service-role key.
begin;
create table public.companies (
 id uuid primary key default gen_random_uuid(), name text not null check(length(trim(name)) between 2 and 120),
 industry text not null default 'Logistics', created_by uuid not null references auth.users(id), created_at timestamptz not null default now()
);
create table public.company_members (
 company_id uuid not null references public.companies(id) on delete cascade,
 user_id uuid not null references auth.users(id) on delete cascade,
 role text not null check(role in ('owner','dispatcher','viewer')), display_name text not null default '', primary key(company_id,user_id)
);
create table public.company_invites (
 id uuid primary key default gen_random_uuid(), company_id uuid not null references public.companies(id) on delete cascade,
 email text not null check(email=lower(trim(email)) and email like '%@%'),
 role text not null check(role in ('dispatcher','viewer')), created_at timestamptz not null default now(), unique(company_id,email)
);
create table public.workspace_records (
 id uuid primary key, company_id uuid not null references public.companies(id) on delete cascade,
 kind text not null check(kind in ('scenario','run')), payload jsonb not null check(jsonb_typeof(payload)='object'), updated_at timestamptz not null default now()
);
create index workspace_records_company on public.workspace_records(company_id,kind);
create index company_members_user on public.company_members(user_id);
create function public.workspace_role(target uuid) returns text language sql stable security definer set search_path='' as $$
 select role from public.company_members where company_id=target and user_id=auth.uid();
$$;
revoke all on function public.workspace_role(uuid) from public;
grant execute on function public.workspace_role(uuid) to authenticated;
alter table public.companies enable row level security;
alter table public.company_members enable row level security;
alter table public.company_invites enable row level security;
alter table public.workspace_records enable row level security;
create policy company_read on public.companies for select to authenticated using(public.workspace_role(id) is not null);
create policy company_update on public.companies for update to authenticated using(public.workspace_role(id)='owner') with check(public.workspace_role(id)='owner');
create policy member_read on public.company_members for select to authenticated using(public.workspace_role(company_id) is not null);
create policy invite_read on public.company_invites for select to authenticated using(public.workspace_role(company_id)='owner');
create policy invite_create on public.company_invites for insert to authenticated with check(public.workspace_role(company_id)='owner');
create policy invite_delete on public.company_invites for delete to authenticated using(public.workspace_role(company_id)='owner');
create policy record_read on public.workspace_records for select to authenticated using(public.workspace_role(company_id) is not null);
create policy record_insert on public.workspace_records for insert to authenticated with check(public.workspace_role(company_id) in ('owner','dispatcher'));
create policy record_update on public.workspace_records for update to authenticated using(public.workspace_role(company_id) in ('owner','dispatcher')) with check(public.workspace_role(company_id) in ('owner','dispatcher'));
create policy record_delete on public.workspace_records for delete to authenticated using(public.workspace_role(company_id) in ('owner','dispatcher'));
create function public.create_company(company_name text,company_industry text default 'Logistics') returns uuid language plpgsql security definer set search_path='' as $$
declare company uuid;
begin
 if auth.uid() is null then raise exception 'Sign in first'; end if;
 insert into public.companies(name,industry,created_by) values(trim(company_name),company_industry,auth.uid()) returning id into company;
 insert into public.company_members(company_id,user_id,role,display_name) values(company,auth.uid(),'owner',coalesce(auth.jwt()->'user_metadata'->>'full_name','Owner'));
 return company;
end;
$$;
create function public.accept_company_invites() returns void language plpgsql security definer set search_path='' as $$
declare verified_email text;
begin
 select lower(email) into verified_email from auth.users where id=auth.uid() and email_confirmed_at is not null;
 if verified_email is null then return; end if;
 insert into public.company_members(company_id,user_id,role,display_name)
 select company_id,auth.uid(),role,coalesce(auth.jwt()->'user_metadata'->>'full_name','Team member') from public.company_invites where email=verified_email
 on conflict(company_id,user_id) do nothing;
 delete from public.company_invites where email=verified_email;
end;
$$;
create function public.set_member_role(target_company uuid,target_user uuid,new_role text) returns void language plpgsql security definer set search_path='' as $$
begin
 if public.workspace_role(target_company) is distinct from 'owner' then raise exception 'Owner access required'; end if;
 if new_role not in ('dispatcher','viewer') then raise exception 'Invalid role'; end if;
 update public.company_members set role=new_role where company_id=target_company and user_id=target_user and role<>'owner';
end;
$$;
revoke all on function public.create_company(text,text) from public;
revoke all on function public.accept_company_invites() from public;
revoke all on function public.set_member_role(uuid,uuid,text) from public;
grant execute on function public.create_company(text,text) to authenticated;
grant execute on function public.accept_company_invites() to authenticated;
grant execute on function public.set_member_role(uuid,uuid,text) to authenticated;
revoke all on public.companies,public.company_members,public.company_invites,public.workspace_records from anon,authenticated;
grant select,update(name,industry) on public.companies to authenticated;
grant select on public.company_members to authenticated;
grant select,insert,delete on public.company_invites to authenticated;
grant select,insert,update,delete on public.workspace_records to authenticated;
commit;
