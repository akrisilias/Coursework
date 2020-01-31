import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.productsService';
import { Router } from '@angular/router';

@Component({
  selector: 'app-logout',
  templateUrl: './logout.component.html',
  styleUrls: ['./logout.component.scss']
})
export class LogoutComponent implements OnInit {

  constructor(private data: DataService, private router: Router) { }

  ngOnInit() {
  }

  postClick() {
    this.data.postLogout()
      .subscribe( data => {
        this.data.auth = "";
        sessionStorage.setItem("auth", "");
        sessionStorage.setItem("user", "");
        this.router.navigate(['']);
      });
  }

}
